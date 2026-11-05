"""Train only the photo frontend for 100 additional label-supervised epochs."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from paths import APP_ASSETS, DATASETS, RESULTS
from .continue_capacity import _EpochPairs, _load_sources, _atomic_json, _atomic_checkpoint
from .micro_capacity import _evaluation_pair, TEST_CONDITIONS, SEED
from .evaluate_rects import _crop_character
from .recognition_guided import (initialize_frontend, load_frozen_classifier, file_hash,
                                FRONTEND_CHECKPOINT, CLASSIFIER_CHECKPOINT)

OUTPUT = RESULTS / 'photo-recognition-guided-100-v1'


class _LabeledPairs(Dataset):
    def __init__(self, images, records, epoch):
        self.pairs = _EpochPairs(images, epoch, .5)
        self.labels = [int(row['class_id']) for row in records]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        photo, _ = self.pairs[index]
        return photo, self.labels[index]


def _metrics(rows):
    return dict(n=len(rows), **{f'top{k}':sum(r['rank']<=k for r in rows)/len(rows) for k in (1,5,10)})


@torch.inference_mode()
def _evaluate(frontend, classifier, tasks, classes, output=None):
    frontend.eval()
    rows=[]
    for start in range(0,len(tasks),16):
        batch=tasks[start:start+16]
        photos=np.stack([p for p,_ in batch])
        normalized=frontend(torch.from_numpy(photos).to('mps'))
        scores=classifier(normalized).float().cpu().numpy()
        for (_,metadata),score in zip(batch,scores):
            order=np.argsort(-score)
            rows.append(dict(**metadata,rank=int(np.flatnonzero(order==metadata['label'])[0])+1,
                             top10=[classes[i] for i in order[:10]]))
    result=dict(metrics=_metrics(rows),rows=rows)
    if output:_atomic_json(output,result)
    return result['metrics']


def _synthetic_tasks(data,records,split):
    base=SEED+(5_000_000 if split=='validation' else 9_000_000)
    return [(_evaluation_pair(clean,base+i*31+j,c,.5)[0],
             dict(sample_id=records[split][i]['sample_id'],label=int(records[split][i]['class_id']),condition=c))
            for i,clean in enumerate(data[split]) for j,c in enumerate(TEST_CONDITIONS)]


def _rects_tasks(classes):
    label_ids={c:i for i,c in enumerate(classes)}
    previous=RESULTS/'photo-restoration-rects-500-v1/predictions.jsonl'
    records=[json.loads(line) for line in previous.read_text().splitlines()]
    tasks=[]
    for row in records:
        with Image.open(DATASETS/'rects-500/img'/f"{row['photo_id']}.jpg") as image:
            crop=_crop_character(image.convert('RGB'),row['points'])
        photo=np.asarray(crop,dtype=np.float32).transpose(2,0,1).copy()/255
        tasks.append((photo,dict(photo_id=row['photo_id'],index=row['index'],character=row['character'],label=label_ids[row['character']])))
    assert len(tasks)==4449
    return tasks


def _report(output,settings,history,evaluations,state):
    lines=['# 한자 정답으로 전처리 모델만 학습','',f'상태: {state}', '',
      '기존 50자·1,000개 원본(손글씨 638, 글꼴 362)과 동일한 분할. ReCTS는 학습·모델 선택에 사용하지 않는다.',
      '절반 굵기 복원 100회 체크포인트에서 시작하여 한자 정답 cross-entropy만으로 추가 100회 학습. 픽셀 정답 손실 없음.',
      '현재 앱과 동일한 4,998자 인식기를 고정하고 인식기를 통해 입력으로 역전파한다. AdamW는 전처리 모델만 받는다.',
      '전처리 출력은 연속값 sigmoid→128px bilinear resize→[-1,1]. 학습과 평가에 동일 경로를 사용하며 이진화·자동 bounding-box crop을 적용하지 않는다.',
      '따라서 0회 결과도 이전 2.6%와 입력 연결 방식이 다르다. before/after의 같은 연속값 경로 비교를 함께 기록한다.',
      '', '|epoch|검증 Top-1|검증 Top-10|','|---|---:|---:|']
    for e in evaluations:
        lines.append(f"|{e['epoch']}|{e['validation']['top1']:.2%}|{e['validation']['top10']:.2%}|")
    lines+=['',f'완료한 추가 학습: {len(history)}회. 검증은 10회마다 기록. 최종 평가는 고정된 100회 모델을 사용하며 ReCTS 성적으로 설정을 조정하지 않는다.',
      '', '학습 중 생성하는 합성 데이터에는 색·배경·조명·절반 굵기·흐림 변형을 사용한다. 글자 종류가 50자이므로 전체 4,998자 학습 결과가 아니다.',
      '', '재개: `PYTHONPATH=tools/handwriting .venv/bin/python -m photo_restoration.train_recognition_guided`']
    if (output/'result.json').exists():
        result=json.loads((output/'result.json').read_text())
        lines+=['','## ReCTS 500장, 같은 4,449개 한자','','|방법|Top-1|Top-5|Top-10|','|---|---:|---:|---:|']
        for name,m in [('자동 반전·이진화',result['binary_baseline']),('연속값 연결 초기 모델',result['before_rects']),('정답 지도 100회 모델',result['after_rects'])]:
            lines.append(f"|{name}|{m['top1']:.2%}|{m['top5']:.2%}|{m['top10']:.2%}|")
        lines+=['','ReCTS는 반복 관찰한 진단 세트이며, 독립적인 최종 실사 성능 추정은 아니다. 정답 글자 위치를 사용했으며 자동 검출 평가는 아니다.']
    (output/'REPORT.md').write_text('\n'.join(lines))


def run(output):
    torch.set_num_threads(4);torch.manual_seed(SEED)
    data,records=_load_sources()
    classifier,classes=load_frozen_classifier('mps')
    original={k:v.detach().cpu().clone() for k,v in classifier.state_dict().items()}
    frontend=initialize_frontend('mps')
    optimizer=torch.optim.AdamW(frontend.parameters(),lr=1e-4,weight_decay=1e-4)
    settings=dict(epochs=100,seed=SEED,train_sources=1000,train_classes=50,batch_size=16,learning_rate=1e-4,
      loss='character cross entropy only',classifier_frozen=True,rects_training=False,
      classifier_sha256=file_hash(CLASSIFIER_CHECKPOINT),frontend_initial_sha256=file_hash(FRONTEND_CHECKPOINT),
      app_onnx_sha256=file_hash(APP_ASSETS/'hanja.onnx'),split_sha256=file_hash(RESULTS/'photo-restoration-micro-capacity-v1/split.json'))
    for split in records:
        assert all(classes[int(r['class_id'])]==r['character'] for r in records[split])
    assert len({r['character'] for r in records['train']})==50
    epoch=0;history=[];evaluations=[]
    if (output/'last.pt').exists():
        saved=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        assert saved['settings']==settings
        frontend.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        epoch=saved['epoch'];history=saved['history'];evaluations=saved['evaluations']
        torch.set_rng_state(saved['torch_rng'])
    _atomic_json(output/'settings.json',settings)
    _atomic_json(output/'runtime.json',dict(pid=os.getpid(),started_at=time.time(),resumed_epoch=epoch))
    validation=_synthetic_tasks(data,records,'validation')
    if not (output/'before-rects.json').exists():
        _atomic_json(output/'progress.json',dict(state='baseline_evaluation',epoch=epoch))
        tasks=_rects_tasks(classes)
        _evaluate(frontend,classifier,tasks,classes,output/'before-rects.json')
        del tasks
    def save():
        _atomic_checkpoint(output/'last.pt',dict(model={k:v.detach().cpu() for k,v in frontend.state_dict().items()},
            optimizer=optimizer.state_dict(),epoch=epoch,settings=settings,history=history,evaluations=evaluations,torch_rng=torch.get_rng_state()))
    while True:
        if epoch%10==0 and not any(e['epoch']==epoch for e in evaluations):
            metric=_evaluate(frontend,classifier,validation,classes,output/f'validation-{epoch:03d}.json')
            evaluations.append(dict(epoch=epoch,validation=metric));save()
            _atomic_json(output/'evaluations.json',evaluations)
            print('VALIDATION',epoch,metric,flush=True)
        if epoch==100:break
        epoch+=1;frontend.train();started=time.time()
        dataset=_LabeledPairs(data['train'],records['train'],epoch+100)
        order=np.random.default_rng(SEED+epoch).permutation(len(dataset)).tolist()
        loader=DataLoader(dataset,batch_size=16,sampler=order,num_workers=2,multiprocessing_context='spawn')
        loss_sum=0.;correct=0;count=0
        for step,(photo,label) in enumerate(loader,1):
            photo,label=photo.to('mps'),label.to('mps')
            optimizer.zero_grad(set_to_none=True)
            logits=classifier(frontend(photo))
            loss=F.cross_entropy(logits,label)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(frontend.parameters(),1.,error_if_nonfinite=True)
            if epoch==1 and step==1:
                assert norm.item()>0 and all(p.grad is None for p in classifier.parameters())
                _atomic_json(output/'gradient-check.json',dict(frontend_gradient_norm=norm.item(),classifier_gradients_absent=True))
            optimizer.step();n=len(label);count+=n;loss_sum+=loss.item()*n
            correct+=(logits.argmax(1)==label).sum().item()
            if step%16==0:
                _atomic_json(output/'progress.json',dict(state='training',epoch=epoch,step=step,steps=len(loader),loss=loss_sum/count))
        history.append(dict(epoch=epoch,loss=loss_sum/count,train_top1=correct/count,seconds=time.time()-started))
        save();_atomic_json(output/'history.json',history)
        remaining=(100-epoch)*float(np.mean([h['seconds'] for h in history[-5:]]))
        _atomic_json(output/'progress.json',dict(state='training',completed_epoch=epoch,epoch=epoch,max_epochs=100,remaining_training_seconds=remaining))
        _report(output,settings,history,evaluations,'학습 중')
        print('EPOCH',history[-1],flush=True)
    _atomic_json(output/'progress.json',dict(state='final_evaluation',epoch=100))
    synthetic=_evaluate(frontend,classifier,_synthetic_tasks(data,records,'test'),classes,output/'after-synthetic.json')
    rects=_evaluate(frontend,classifier,_rects_tasks(classes),classes,output/'after-rects.json')
    assert all(torch.equal(v.detach().cpu(),original[k]) for k,v in classifier.state_dict().items())
    assert file_hash(APP_ASSETS/'hanja.onnx')==settings['app_onnx_sha256']
    before=json.loads((output/'before-rects.json').read_text())['metrics']
    baseline=json.loads((RESULTS/'photo-restoration-rects-500-v1/result.json').read_text())['metrics']['auto_binary']
    _atomic_json(output/'result.json',dict(epoch=100,before_rects=before,after_rects=rects,after_synthetic=synthetic,
        binary_baseline=baseline,classifier_unchanged=True,training_seconds=sum(h['seconds'] for h in history)))
    _report(output,settings,history,evaluations,'완료')
    _atomic_json(output/'progress.json',dict(state='complete',epoch=100))
    print('COMPLETE',rects,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args.output)
        except BaseException as error:
            _atomic_json(args.output/'error.json',dict(error=str(error),traceback=traceback.format_exc()))
            _atomic_json(args.output/'progress.json',dict(state='error',error=str(error)))
            raise


if __name__=='__main__':main()
