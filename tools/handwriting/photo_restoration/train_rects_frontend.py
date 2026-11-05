"""Train a photo frontend on independent ReCTS photos; freeze the recognizer."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from paths import APP_ASSETS, RESULTS
from .continue_capacity import _atomic_json, _atomic_checkpoint
from .recognition_guided import initialize_frontend, load_frozen_classifier, file_hash, CLASSIFIER_CHECKPOINT
from .rects_training_data import DATA, HELD_OUT, SEED, ReCTSCharacters, prepare
from .train_recognition_guided import _evaluate, _rects_tasks

OUTPUT=RESULTS/'rects-frontend-500-v1'
INITIAL=RESULTS/'photo-recognition-guided-100-v1/last.pt'


def _improved(score,best,minimum=.001):
    return score > best + minimum


@torch.inference_mode()
def _validate(frontend,classifier,loader):
    frontend.eval()
    counts={1:0,5:0,10:0};total=0;loss_sum=0.
    for rgb,label in loader:
        rgb,label=rgb.to('mps'),label.to('mps')
        logits=classifier(frontend(rgb));order=logits.topk(10,dim=1).indices
        loss_sum+=F.cross_entropy(logits,label,reduction='sum').item();total+=len(label)
        for k in counts:counts[k]+=(order[:,:k]==label[:,None]).any(1).sum().item()
    return dict(n=total,loss=loss_sum/total,**{f'top{k}':v/total for k,v in counts.items()})


def _report(output,settings,history,best,state):
    lines=['# ReCTS 실제 사진으로 전처리만 학습','',f'상태: {state}','',
      '기존 한자 정답 지도 25회 전처리 체크포인트에서 시작. 현재 4,998자 인식기는 고정하며 전처리 모델만 업데이트한다.',
      '새로 받은 사진 500장을 원본 사진 단위로 80:20 학습/검증 분할. 기존 평가 500장과 ID 및 동일 RGB 픽셀 중복을 제외한다. 유사 장면의 독립성까지 보장하지는 않는다.',
      f"분할: {settings['data_summary']}",
      '한자 정답 cross entropy만 사용. 연속값 출력→128px resize→고정 인식기 경로를 학습과 평가에서 동일하게 사용. 추가 이미지 변형 없음.',
      '최대 100회, 매회 검증. 검증 Top-1이 기존 최선보다 0.1%p 넘게 개선되지 않은 회차가 10회 연속이면 종료. 학습 전 모델도 best 후보다. 평가 500장은 조기 종료·모델 선택에 사용하지 않는다.',
      '',f"최선의 검증 회차: {best['epoch']} / Top-1 {best['top1']:.2%}",'',
      '|epoch|학습 loss|학습 Top-1|검증 Top-1|검증 Top-10|','|---|---:|---:|---:|---:|']
    for row in history:
        lines.append(f"|{row['epoch']}|{row['loss']:.4f}|{row['train_top1']:.2%}|{row['validation']['top1']:.2%}|{row['validation']['top10']:.2%}|")
    if (output/'result.json').exists():
        r=json.loads((output/'result.json').read_text())
        lines+=['','## 기존 평가 사진 500장 / 같은 한자 4,449개','','|방법|Top-1|Top-5|Top-10|','|---|---:|---:|---:|']
        for name,key in [('자동 반전·이진화','binary_baseline'),('ReCTS 학습 전 25회 모델','before_rects'),('ReCTS 학습 후 검증 최선 모델','after_rects')]:
            m=r[key];lines.append(f"|{name}|{m['top1']:.2%}|{m['top5']:.2%}|{m['top10']:.2%}|")
        lines+=['','정답 글자 위치를 사용한 인식 평가. 반복 관찰한 평가 자료이므로 독립적인 최종 실사 성능 추정은 아니다.']
    (output/'REPORT.md').write_text('\n'.join(lines))


def run(output):
    if (output/'result.json').exists():
        print('Already complete',flush=True);return
    torch.set_num_threads(4);torch.manual_seed(SEED)
    classifier,classes=load_frozen_classifier('mps')
    frozen={k:v.detach().cpu().clone() for k,v in classifier.state_dict().items()}
    initial=torch.load(INITIAL,map_location='cpu',weights_only=False)
    assert initial['epoch']==25
    frontend=initialize_frontend('mps');frontend.load_state_dict(initial['model'])
    optimizer=torch.optim.AdamW(frontend.parameters(),lr=1e-4,weight_decay=1e-4)
    data_hash=file_hash(DATA/'manifest.json');test_hash=file_hash(HELD_OUT/'manifest.json')
    if (output/'split.json').exists():split=json.loads((output/'split.json').read_text())
    else:split=prepare(output,classes)
    settings=dict(max_epochs=100,patience=10,min_delta=.001,seed=SEED,batch_size=16,learning_rate=1e-4,
        initialization_sha256=file_hash(INITIAL),classifier_sha256=file_hash(CLASSIFIER_CHECKPOINT),
        onnx_sha256=file_hash(APP_ASSETS/'hanja.onnx'),data_manifest_sha256=data_hash,test_manifest_sha256=test_hash,
        split_sha256=file_hash(output/'split.json'),data_summary=split['summary'],classifier_frozen=True,
        objective='character cross entropy',evaluation_policy='best validation checkpoint only; no held-out test during training')
    _atomic_json(output/'runtime.json',dict(pid=os.getpid(),started_at=time.time()))
    datasets={s:ReCTSCharacters(output/f'{s}-rgb.npy',split['records'][s]) for s in ('train','validation')}
    validation=DataLoader(datasets['validation'],batch_size=16,shuffle=False,num_workers=0)
    epoch=0;history=[];bad_epochs=0
    if (output/'last.pt').exists():
        saved=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        assert saved['settings']==settings
        frontend.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        epoch=saved['epoch'];history=saved['history'];best=saved['best'];bad_epochs=saved['bad_epochs']
        torch.set_rng_state(saved['torch_rng'])
    else:
        baseline=_validate(frontend,classifier,validation)
        best=dict(epoch=0,top1=baseline['top1'],validation=baseline)
        _atomic_json(output/'initial-validation.json',baseline)
        _atomic_checkpoint(output/'best.pt',dict(model={k:v.detach().cpu() for k,v in frontend.state_dict().items()},epoch=0,settings=settings))
    _atomic_json(output/'settings.json',settings)
    def save():
        _atomic_checkpoint(output/'last.pt',dict(model={k:v.detach().cpu() for k,v in frontend.state_dict().items()},optimizer=optimizer.state_dict(),
           epoch=epoch,history=history,best=best,bad_epochs=bad_epochs,settings=settings,torch_rng=torch.get_rng_state()))
    save();print('SETTINGS',json.dumps(settings),flush=True)
    while epoch<100 and bad_epochs<10:
        epoch+=1;started=time.time();frontend.train()
        loader=DataLoader(datasets['train'],batch_size=16,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(SEED+epoch))
        total=0;loss_sum=0.;correct=0
        for step,(rgb,label) in enumerate(loader,1):
            rgb,label=rgb.to('mps'),label.to('mps');optimizer.zero_grad(set_to_none=True)
            logits=classifier(frontend(rgb));loss=F.cross_entropy(logits,label)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(frontend.parameters(),1.,error_if_nonfinite=True)
            if epoch==1 and step==1:
                assert norm.item()>0 and all(p.grad is None for p in classifier.parameters())
                _atomic_json(output/'gradient-check.json',dict(frontend_gradient_norm=norm.item(),classifier_gradients_absent=True))
            optimizer.step();total+=len(label);loss_sum+=loss.item()*len(label);correct+=(logits.argmax(1)==label).sum().item()
            if step==1 or step%32==0:
                _atomic_json(output/'progress.json',dict(state='training',epoch=epoch,step=step,steps=len(loader),loss=loss_sum/total))
        metrics=_validate(frontend,classifier,validation)
        if _improved(metrics['top1'],best['top1']):
            best=dict(epoch=epoch,top1=metrics['top1'],validation=metrics);bad_epochs=0
            _atomic_checkpoint(output/'best.pt',dict(model={k:v.detach().cpu() for k,v in frontend.state_dict().items()},epoch=epoch,settings=settings))
        else:bad_epochs+=1
        history.append(dict(epoch=epoch,loss=loss_sum/total,train_top1=correct/total,validation=metrics,seconds=time.time()-started))
        assert all(p.grad is None for p in classifier.parameters())
        save();_atomic_json(output/'history.json',history)
        remaining=(100-epoch)*float(np.mean([r['seconds'] for r in history[-5:]]))
        _atomic_json(output/'progress.json',dict(state='training',completed_epoch=epoch,max_epochs=100,best=best,bad_epochs=bad_epochs,maximum_remaining_seconds=remaining))
        _report(output,settings,history,best,'학습 중');print('EPOCH',json.dumps(history[-1]),flush=True)
    _atomic_json(output/'progress.json',dict(state='final_evaluation',epoch=epoch,best=best))
    chosen=torch.load(output/'best.pt',map_location='cpu',weights_only=False);frontend.load_state_dict(chosen['model'])
    after=_evaluate(frontend,classifier,_rects_tasks(classes),classes,output/'after-rects.json')
    assert all(torch.equal(v.detach().cpu(),frozen[k]) for k,v in classifier.state_dict().items())
    assert file_hash(APP_ASSETS/'hanja.onnx')==settings['onnx_sha256']
    before=json.loads((RESULTS/'photo-recognition-guided-100-v1/epoch-025-rects/result.json').read_text())
    assert before['checkpoint_sha256']==settings['initialization_sha256']
    result=dict(completed_epoch=epoch,best_epoch=chosen['epoch'],stopping_reason='validation_plateau' if bad_epochs>=10 else 'max_epochs',
        before_rects=before['metrics'],after_rects=after,binary_baseline=before['binary'],classifier_unchanged=True,
        training_seconds=sum(r['seconds'] for r in history),data_summary=split['summary'])
    _atomic_json(output/'result.json',result);_report(output,settings,history,best,'완료')
    _atomic_json(output/'progress.json',dict(state='complete',epoch=epoch,best_epoch=chosen['epoch']))
    print('COMPLETE',json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args.output)
        except BaseException as error:
            _atomic_json(args.output/'error.json',dict(error=str(error),traceback=traceback.format_exc()))
            _atomic_json(args.output/'progress.json',dict(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'error',error=str(error)))
            raise


if __name__=='__main__':main()
