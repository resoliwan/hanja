"""Train only GlyphWeave's 3500-class cosine head; cache frozen features first."""
from paths import DATASETS, MODELS, RESULTS, ROOT
import argparse,collections,csv,hashlib,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import ConcatDataset,DataLoader,Subset


sys.path.insert(0,str(ROOT/'vendor/GlyphWeave'))
from handwriting.model import build_model,CosineHead
from handwriting.images import GlyphTransform
from handwriting.runtime import load_checkpoint,save_checkpoint
from hanja3500_dataset import Hanja3500Dataset

DATA=DATASETS / 'korean-hanja3500-v1'
RUN=RESULTS / 'hanja3500-head-only-v1'
ORIGINAL=MODELS / 'glyphweave-v1.0.0-inference/best.pt'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()

def cpu_state(module):return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}

def metrics(labels,top):
    if not len(labels):return dict(n=0,classes=0,top1=None,top5=None,macro_top1=None)
    ok=top[:,0]==labels;ok5=(top==labels[:,None]).any(1)
    totals=np.bincount(labels,minlength=3500);right=np.bincount(labels,weights=ok,minlength=3500)
    return dict(n=len(labels),classes=int((totals>0).sum()),top1=float(ok.mean()),top5=float(ok5.mean()),
                macro_top1=float((right[totals>0]/totals[totals>0]).mean()))

@torch.inference_mode()
def predict(head,features,indices,device,batch=1024):
    head.eval();result=[]
    for start in range(0,len(indices),batch):
        x=torch.from_numpy(np.array(features[indices[start:start+batch]],copy=True)).to(device)
        result.append(head(x).topk(5,dim=1).indices.cpu().numpy())
    return np.concatenate(result)

def evaluate(head,features,indices,rows,labels,classes,old_chars,grades,device,prefix):
    top=predict(head,features,indices,device);ys=labels[indices]
    masks={
        'font_base':np.array([r['source']=='fonts' and r['variant'].endswith(':base') for r in rows]),
        'font_variants':np.array([r['source']=='fonts' and not r['variant'].endswith(':base') for r in rows]),
        'scut_handwriting':np.array([r['source']=='scut' for r in rows]),
    }
    summary={}
    known=np.array([r['character'] in old_chars for r in rows])
    for name,mask in masks.items():
        summary[name]=metrics(ys[mask],top[mask])
        summary[name]['existing_characters']=metrics(ys[mask&known],top[mask&known])
        summary[name]['new_characters']=metrics(ys[mask&~known],top[mask&~known])
        summary[name]['by_grade']={g:metrics(ys[mask&np.array([r['grade']==g for r in rows])],top[mask&np.array([r['grade']==g for r in rows])]) for g in grades}
    with (RUN/(prefix+'-predictions.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['sample_id','character','source','variant','grade','top1','top5','candidates'])
        for row,ids,y in zip(rows,top,ys):
            w.writerow([row['sample_id'],row['character'],row['source'],row['variant'],row['grade'],bool(ids[0]==y),bool(y in ids),' '.join(classes[int(i)] for i in ids)])
    with (RUN/(prefix+'-per-character.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['evaluation','character','samples','correct_top1','correct_top5'])
        for name,mask in masks.items():
            total=np.bincount(ys[mask],minlength=3500)
            ok=np.bincount(ys[mask],weights=(top[mask,0]==ys[mask]),minlength=3500)
            ok5=np.bincount(ys[mask],weights=(top[mask]==ys[mask,None]).any(1),minlength=3500)
            for i,n in enumerate(total):
                if n:w.writerow([name,classes[i],int(n),int(ok[i]),int(ok5[i])])
    (RUN/(prefix+'-evaluation.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(prefix,json.dumps({k:{j:v for j,v in s.items() if j in ['top1','top5','macro_top1']} for k,s in summary.items()}),flush=True)
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);p.add_argument('--feature-batch',type=int,default=64);a=p.parse_args()
    RUN.mkdir(exist_ok=True);start_time=time.time()
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
    assert torch.backends.mps.is_available();device=torch.device('mps')
    transform=GlyphTransform(128,training=False)
    candidates=Hanja3500Dataset(DATA,'train',transform=transform)
    evaluation=Hanja3500Dataset(DATA,'evaluation',transform=transform)
    rows=candidates.rows+evaluation.rows;classes=candidates.classes
    labels=np.array([int(r['class_id']) for r in rows],dtype=np.int64)
    valid_ids=set((RESULTS / 'hanja3500-training-plan/proposed-validation-sample-ids.txt').read_text().splitlines())
    val_idx=np.array([i for i,r in enumerate(candidates.rows) if r['sample_id'] in valid_ids],dtype=np.int64)
    train_idx=np.array([i for i,r in enumerate(candidates.rows) if r['sample_id'] not in valid_ids],dtype=np.int64)
    eval_idx=np.arange(len(candidates),len(rows),dtype=np.int64)
    assert len(train_idx)==226527 and len(val_idx)==14103 and len(eval_idx)==73987
    assert len(set(labels[train_idx]))==3500 and len(val_idx)==len(valid_ids)
    assert not ({rows[i]['pixel_sha256'] for i in train_idx}&{rows[i]['pixel_sha256'] for i in np.concatenate([val_idx,eval_idx])})
    original=load_checkpoint(ORIGINAL);old_chars=set(original['classes']);old_index={c:i for i,c in enumerate(original['classes'])}
    config=original['config'];model=build_model(config,3500)
    transferred={k:v for k,v in original['ema'].items() if k!='classifier.weight'}
    result=model.load_state_dict(transferred,strict=False);assert result.missing_keys==['classifier.weight'] and not result.unexpected_keys
    with torch.no_grad():
        for i,c in enumerate(classes):
            if c in old_index:model.classifier.weight[i].copy_(original['ema']['classifier.weight'][old_index[c]])
    initial=cpu_state(model)
    for name,param in model.named_parameters():param.requires_grad_(name.startswith('classifier.'))
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==1120001
    settings=dict(experiment='head-only',seed=42,classes=3500,transferred_output_classes=sum(c in old_index for c in classes),
        new_output_classes=sum(c not in old_index for c in classes),trainable_parameters=1120001,frozen_parameters=5501003,
        learning_rate=0.001,weight_decay=0.01,label_smoothing=0.05,batch_size=256,draws_per_epoch=35000,
        max_epochs=10,patience=3,train_samples=len(train_idx),validation_samples=len(val_idx),evaluation_samples=len(eval_idx),
        trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],original_sha256=sha(ORIGINAL),
        data_manifest_sha256={p.name:sha(p) for p in sorted((DATA/'manifests').glob('*.csv'))},
        validation_ids_sha256=sha(RESULTS / 'hanja3500-training-plan/proposed-validation-sample-ids.txt'),
        class_map_sha256=sha(DATA/'classes.txt'),preprocessing='official GlyphTransform(128, training=False)',
        feature_dtype='float32',device='Apple M4 Pro MPS',torch_version=str(torch.__version__))
    (RUN/'settings.json').write_text(json.dumps(settings,indent=2))
    for name,ids in [('train',train_idx),('validation',val_idx),('evaluation',eval_idx)]:
        (RUN/(name+'-sample-ids.txt')).write_text('\n'.join(rows[i]['sample_id'] for i in ids)+'\n')
    def checkpoint(path,head_state,epoch,validation=None,optimizer=None):
        state=dict(initial)
        state.update({'classifier.'+k:v for k,v in head_state.items()})
        payload=dict(config=config,classes=classes,ema=state,model=state,epoch=epoch,experiment=settings,validation=validation)
        if optimizer is not None:payload['optimizer']=optimizer
        save_checkpoint(path,payload)
    checkpoint(RUN/'initial-3500.pt',cpu_state(model.classifier),0)
    head=model.classifier.to(device)
    model.classifier=torch.nn.Identity();model.to(device).eval()
    assert not any(p.requires_grad for p in model.parameters())
    signature=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()+''.join(r['sample_id']+r['pixel_sha256'] for r in rows).encode()).hexdigest()
    progress_path=RUN/'features-progress.json';cache_path=RUN/'features.f32'
    done=0
    if progress_path.exists():
        progress=json.loads(progress_path.read_text());assert progress['signature']==signature
        done=progress['completed'];assert cache_path.stat().st_size==len(rows)*320*4
    features=np.memmap(cache_path,dtype='float32',mode='r+' if cache_path.exists() else 'w+',shape=(len(rows),320))
    ds=ConcatDataset([candidates,evaluation]);extract_start=time.time();initial_done=done
    print('DATA READY',json.dumps({k:settings[k] for k in ['train_samples','validation_samples','evaluation_samples']}),flush=True)
    print('FEATURES',done,'/',len(rows),flush=True)
    loader=DataLoader(Subset(ds,range(done,len(rows))),batch_size=a.feature_batch,num_workers=a.workers,
        shuffle=False,multiprocessing_context='spawn' if a.workers else None,persistent_workers=bool(a.workers))
    next_log=(done//4096+1)*4096
    with torch.inference_mode():
        for images,ys in loader:
            n=len(ys);assert np.array_equal(ys.numpy(),labels[done:done+n])
            x=model(images.to(device)).cpu().numpy();assert x.shape==(n,320) and np.isfinite(x).all()
            features[done:done+n]=x;done+=n
            if done>=next_log or done==len(rows):
                features.flush();progress_path.write_text(json.dumps(dict(signature=signature,completed=done,total=len(rows))))
                speed=(done-initial_done)/max(time.time()-extract_start,0.001)
                print(f'FEATURES {done}/{len(rows)} {speed:.1f} images/sec ETA {(len(rows)-done)/speed/60:.1f} min',flush=True)
                next_log+=4096
    del loader
    # No backbone weights have moved during feature extraction; no optimizer sees them.
    frozen=cpu_state(model)
    assert all(torch.equal(v,initial[k]) for k,v in frozen.items())
    del model;torch.mps.empty_cache()
    extraction_seconds=time.time()-extract_start
    grades=['8','7','6','준5','5','준4','4','준3','3','준2','2','준1','1']
    before=evaluate(head,features,eval_idx,evaluation.rows,labels,classes,old_chars,grades,device,'before')
    initial_val=metrics(labels[val_idx],predict(head,features,val_idx,device))
    (RUN/'initial-validation.json').write_text(json.dumps(initial_val,indent=2))
    print('INITIAL VALIDATION',json.dumps(initial_val),flush=True)
    buckets={i:{'ai-free':[],'fonts':[]} for i in range(3500)}
    for i in train_idx:buckets[int(labels[i])][rows[i]['source']].append(int(i))
    rng=np.random.default_rng(42)
    optimizer=torch.optim.AdamW([dict(params=[head.weight],weight_decay=0.01),dict(params=[head.log_scale],weight_decay=0)],lr=0.001)
    best_score=initial_val['macro_top1'];best_epoch=0;bad=0;history=[]
    checkpoint(RUN/'head-only-best.pt',cpu_state(head),0,initial_val)
    training_start=time.time()
    for epoch in range(1,11):
        epoch_start=time.time();order=np.repeat(np.arange(3500),10);rng.shuffle(order);draws=[]
        for c in order:
            pools=buckets[int(c)];src='ai-free' if pools['ai-free'] and rng.random()<0.75 else 'fonts'
            pool=pools[src] or pools['ai-free'];draws.append(pool[int(rng.integers(len(pool)))])
        draws=np.array(draws,dtype=np.int64);head.train();loss_sum=0;correct=0
        for start in range(0,len(draws),256):
            ix=draws[start:start+256]
            x=torch.from_numpy(np.array(features[ix],copy=True)).to(device);y=torch.from_numpy(labels[ix]).to(device)
            optimizer.zero_grad(set_to_none=True);logits=head(x)
            loss=torch.nn.functional.cross_entropy(logits,y,label_smoothing=0.05)
            assert torch.isfinite(loss).item();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.0,error_if_nonfinite=True);optimizer.step()
            loss_sum+=loss.item()*len(ix);correct+=int((logits.argmax(1)==y).sum().item())
        val=metrics(labels[val_idx],predict(head,features,val_idx,device))
        record=dict(epoch=epoch,draws=len(draws),loss=loss_sum/len(draws),sampled_train_top1=correct/len(draws),validation=val,seconds=time.time()-epoch_start)
        history.append(record)
        with (RUN/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        checkpoint(RUN/'head-only-last.pt',cpu_state(head),epoch,val,optimizer.state_dict())
        if val['macro_top1']>best_score:
            best_score=val['macro_top1'];best_epoch=epoch;bad=0
            checkpoint(RUN/'head-only-best.pt',cpu_state(head),epoch,val)
        else:bad+=1
        print('EPOCH',json.dumps(record),flush=True)
        if bad>=3:print('EARLY STOP',flush=True);break
    best=load_checkpoint(RUN/'head-only-best.pt')
    head.load_state_dict({k.removeprefix('classifier.'):v for k,v in best['ema'].items() if k.startswith('classifier.')})
    after=evaluate(head,features,eval_idx,evaluation.rows,labels,classes,old_chars,grades,device,'after')
    changed=[k for k in initial if not torch.equal(initial[k],best['ema'][k])]
    assert all(k.startswith('classifier.') for k in changed)
    # Check the saved full model is inference-compatible and agrees with cached logits.
    final_model=build_model(config,3500);final_model.load_state_dict(best['ema'],strict=True);final_model.to(device).eval()
    sample_checks=[]
    with torch.inference_mode():
        for i in [0,len(candidates)-1,len(candidates),len(rows)-1]:
            image,y=ds[i];full=final_model(image.unsqueeze(0).to(device));cached=head(torch.from_numpy(np.array(features[i:i+1],copy=True)).to(device))
            error=float((full-cached).abs().max().item())
            assert torch.equal(full.topk(5).indices,cached.topk(5).indices),(i,error)
            sample_checks.append(dict(sample_id=rows[i]['sample_id'],max_logit_difference=error))
    result=dict(status='complete',best_epoch=best_epoch,epochs_completed=len(history),initial_validation=initial_val,
        best_validation=best['validation'],before=before,after=after,changed_state_keys=changed,
        frozen_state_unchanged=True,training_evaluation_hash_overlap=0,checkpoint_inference_checks=sample_checks,
        feature_extraction_seconds=extraction_seconds,training_and_final_evaluation_seconds=time.time()-training_start,
        total_seconds=time.time()-start_time,best_checkpoint_sha256=sha(RUN/'head-only-best.pt'))
    (RUN/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    lines=['# 한국 한자 3,500자 — 출력층만 학습 결과','',f'최우수 회차: {best_epoch}. 총 실행 회차: {len(history)}. 입력 특징 추출 부분 고정, 출력층만 학습.','',
        '| 평가 자료 | 샘플 수 | 학습 전 Top-1 | 학습 후 Top-1 | 학습 전 Top-5 | 학습 후 Top-5 |','|---|---:|---:|---:|---:|---:|']
    for key,title in [('font_base','Noto Sans 원본'),('font_variants','Noto Sans 변형'),('scut_handwriting','SCUT 실제 필기')]:
        b,aft=before[key],after[key]
        lines.append(f'| {title} | {b["n"]:,} | {b["top1"]:.2%} | {aft["top1"]:.2%} | {b["top5"]:.2%} | {aft["top5"]:.2%} |')
    lines+=['','- 학습 전은 기존 모델을 3,500자 출력으로 재구성한 초기 모델이다. 원본 7,247자 모델 점수와 구분한다.',
        '- 모든 비교는 동일한 평가 ID를 사용한다. 최우수 회차는 AI-FREE 검증 macro Top-1만으로 선택했다.',
        '- 폰트 점수와 실제 필기 점수는 합산하지 않는다. SCUT에 없는 428자는 필기 평가 불가.',
        '- AI-FREE 검증은 필기자 분리 보장이 없으며 유사 이미지 중복 여부는 별도로 검증하지 않았다.',
        '- 학습 원본의 AI-FREE/SCUT 이용 조건은 그대로 적용된다.',
        '- 전체 이미지 사전 시각 검사는 하지 않았다. 학습 입력으로 디코딩/전처리하고 특징의 유한성을 확인했다.',
        '- 고정 가중치 불변, 학습·평가 픽셀 해시 분리, 저장 모델과 특징 캐시의 Top-5 일치를 확인했다.',
        '',f'전체 소요: {result["total_seconds"]/60:.1f}분.','',
        '파일: head-only-best.pt / settings.json / history.jsonl / before-evaluation.json / after-evaluation.json / before-predictions.csv / after-predictions.csv / result.json']
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('COMPLETE',json.dumps({'best_epoch':best_epoch,'minutes':result['total_seconds']/60}),flush=True)

if __name__=='__main__':main()
