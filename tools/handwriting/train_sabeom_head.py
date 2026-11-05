"""Expand the trained head and evaluate on untouched handwriting and font data."""
import collections
import csv
import json
import random
import time
import unicodedata

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

import train_hanja3500_head as shared
from hanja3500_dataset import Hanja3500Dataset
from paths import DATASETS, RESULTS

DATA = DATASETS / 'korean-hanja5000-v1'
RUN = RESULTS / 'hanja5000-head-only-v1'
BASE_DATA = DATASETS / 'korean-hanja3500-v1'
BASE_RUN = RESULTS / 'hanja3500-head-only-v1'


def main():
    RUN.mkdir(exist_ok=True)
    shared.RUN = RUN
    started = time.time()
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.set_num_threads(4)
    assert torch.backends.mps.is_available()
    device = torch.device('mps')
    transform = shared.GlyphTransform(128, training=False)
    candidates = Hanja3500Dataset(DATA, 'train', transform=transform)
    evaluation = Hanja3500Dataset(DATA, 'evaluation', transform=transform)
    dataset = ConcatDataset([candidates, evaluation])
    rows = candidates.rows + evaluation.rows
    classes = candidates.classes; nclasses = len(classes)
    labels = np.array([int(r['class_id']) for r in rows], dtype=np.int64)
    valid_ids = set((DATA/'validation-sample-ids.txt').read_text().splitlines())
    val_idx = np.array([i for i,r in enumerate(candidates.rows) if r['sample_id'] in valid_ids])
    train_idx = np.array([i for i,r in enumerate(candidates.rows) if r['sample_id'] not in valid_ids])
    eval_idx = np.arange(len(candidates),len(rows))
    assert len(set(labels[train_idx])) == nclasses == 4998
    assert not {rows[i]['pixel_sha256'] for i in train_idx} & {rows[i]['pixel_sha256'] for i in np.concatenate([val_idx,eval_idx])}
    base = shared.load_checkpoint(BASE_RUN/'head-only-best.pt')
    original = shared.load_checkpoint(shared.ORIGINAL)
    for key,value in base['ema'].items():
        if not key.startswith('classifier.'):
            assert torch.equal(value,original['ema'][key]),key
    model = shared.build_model(base['config'],nclasses)
    missing = model.load_state_dict({k:v for k,v in base['ema'].items() if k!='classifier.weight'},strict=False)
    assert missing.missing_keys == ['classifier.weight'] and not missing.unexpected_keys
    old_index = {c:i for i,c in enumerate(base['classes'])}
    original_index = {unicodedata.normalize('NFC',c):i for i,c in enumerate(original['classes'])}
    with torch.no_grad():
        for i,c in enumerate(classes):
            if c in old_index:
                model.classifier.weight[i].copy_(base['ema']['classifier.weight'][old_index[c]])
            elif c in original_index:
                model.classifier.weight[i].copy_(original['ema']['classifier.weight'][original_index[c]])
    initial = shared.cpu_state(model)
    for name,param in model.named_parameters():
        param.requires_grad_(name.startswith('classifier.'))
    settings = dict(classes=nclasses,seed=42,initial_checkpoint_sha256=shared.sha(BASE_RUN/'head-only-best.pt'),
                    original_checkpoint_sha256=shared.sha(shared.ORIGINAL),
                    manifests={p.name:shared.sha(p) for p in sorted((DATA/'manifests').glob('*.csv'))},
                    validation_sha256=shared.sha(DATA/'validation-sample-ids.txt'),
                    old_classes=len(old_index),added_classes=nclasses-len(old_index),
                    train=len(train_idx),validation=len(val_idx),evaluation=len(eval_idx),
                    learning_rate=.001,weight_decay=.01,label_smoothing=.05,batch_size=256,
                    draws_per_epoch=nclasses*10,max_epochs=10,patience=3,
                    preprocessing='official GlyphTransform(128, training=False)',
                    training='classifier only; frozen backbone verified identical to 3500 feature cache')
    (RUN/'settings.json').write_text(json.dumps(settings,indent=2))
    for name,ids in [('train',train_idx),('validation',val_idx),('evaluation',eval_idx)]:
        (RUN/(name+'-sample-ids.txt')).write_text('\n'.join(rows[i]['sample_id'] for i in ids)+'\n')
    def checkpoint(name,epoch,validation):
        state=dict(initial)
        state.update({'classifier.'+k:v for k,v in shared.cpu_state(head).items()})
        shared.save_checkpoint(RUN/name,dict(config=base['config'],classes=classes,ema=state,model=state,
                                            epoch=epoch,validation=validation,experiment=settings))
    head=model.classifier.to(device)
    model.classifier=torch.nn.Identity(); model.to(device).eval()
    assert not any(p.requires_grad for p in model.parameters())
    old_train=Hanja3500Dataset(BASE_DATA,'train'); old_eval=Hanja3500Dataset(BASE_DATA,'evaluation')
    old_rows=old_train.rows+old_eval.rows
    old_progress=json.loads((BASE_RUN/'features-progress.json').read_text())
    assert old_progress['completed']==len(old_rows)==old_progress['total']
    old_settings=json.loads((BASE_RUN/'settings.json').read_text())
    assert old_settings['original_sha256']==shared.sha(shared.ORIGINAL)
    for path in (BASE_DATA/'manifests').glob('*.csv'):
        assert shared.sha(path)==old_settings['data_manifest_sha256'][path.name]
    old_features=np.memmap(BASE_RUN/'features.f32',dtype='float32',mode='r',shape=(len(old_rows),320))
    old_lookup={r['pixel_sha256']:i for i,r in enumerate(old_rows)}
    missing_indices=np.array([i for i,r in enumerate(rows) if r['pixel_sha256'] not in old_lookup])
    cache=RUN/'features.f32'; progress_path=RUN/'features-progress.json'
    signature=shared.hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    features=np.memmap(cache,dtype='float32',mode='r+' if cache.exists() else 'w+',shape=(len(rows),320))
    for i,r in enumerate(rows):
        if r['pixel_sha256'] in old_lookup:
            j=old_lookup[r['pixel_sha256']]
            assert old_rows[j]['character']==r['character']
            features[i]=old_features[j]
    done=0
    if progress_path.exists():
        progress=json.loads(progress_path.read_text()); assert progress['signature']==signature
        done=progress['completed_new']
    print('DATA',json.dumps(settings),'REUSED FEATURES',len(rows)-len(missing_indices),'NEW',len(missing_indices),flush=True)
    loader=DataLoader(Subset(dataset,missing_indices[done:].tolist()),batch_size=64,num_workers=2,
                      multiprocessing_context='spawn',persistent_workers=True)
    extract_start=time.time(); initial_done=done; next_log=done+4096
    with torch.inference_mode():
        for images,ys in loader:
            ix=missing_indices[done:done+len(ys)]
            assert np.array_equal(ys.numpy(),labels[ix])
            x=model(images.to(device)).cpu().numpy(); assert np.isfinite(x).all()
            features[ix]=x; done+=len(ix)
            if done>=next_log or done==len(missing_indices):
                features.flush();progress_path.write_text(json.dumps(dict(signature=signature,completed_new=done,total_new=len(missing_indices))))
                speed=(done-initial_done)/max(time.time()-extract_start,.001)
                print(f'FEATURES {done}/{len(missing_indices)} {speed:.1f}/sec ETA {(len(missing_indices)-done)/speed/60:.1f} min',flush=True)
                next_log=done+4096
    del loader
    assert all(torch.equal(v,initial[k]) for k,v in shared.cpu_state(model).items())
    del model; torch.mps.empty_cache()
    grades=sorted({r['grade'] for r in rows})
    before=shared.evaluate(head,features,eval_idx,evaluation.rows,labels,classes,set(old_index),grades,device,'before')
    initial_val=shared.metrics(labels[val_idx],shared.predict(head,features,val_idx,device))
    checkpoint('initial-4998.pt',0,initial_val)
    checkpoint('head-only-best.pt',0,initial_val)
    buckets={i:{'ai-free':[],'fonts':[]} for i in range(nclasses)}
    for i in train_idx:buckets[int(labels[i])][rows[i]['source']].append(int(i))
    rng=np.random.default_rng(42)
    optimizer=torch.optim.AdamW([dict(params=[head.weight],weight_decay=.01),dict(params=[head.log_scale],weight_decay=0)],lr=.001)
    best_score=initial_val['macro_top1'];best_epoch=0;bad=0;history=[]
    (RUN/'history.jsonl').write_text('')
    for epoch in range(1,11):
        order=np.repeat(np.arange(nclasses),10);rng.shuffle(order);draws=[]
        for c in order:
            pools=buckets[int(c)];source='ai-free' if pools['ai-free'] and rng.random()<.75 else 'fonts'
            pool=pools[source] or pools['ai-free'];draws.append(pool[int(rng.integers(len(pool)))])
        draws=np.array(draws);head.train();loss_sum=0
        for start in range(0,len(draws),256):
            ix=draws[start:start+256]
            x=torch.from_numpy(np.array(features[ix],copy=True)).to(device);y=torch.from_numpy(labels[ix]).to(device)
            optimizer.zero_grad(set_to_none=True);logits=head(x)
            loss=torch.nn.functional.cross_entropy(logits,y,label_smoothing=.05)
            assert torch.isfinite(loss).item();loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            loss_sum+=loss.item()*len(ix)
        val=shared.metrics(labels[val_idx],shared.predict(head,features,val_idx,device))
        record=dict(epoch=epoch,loss=loss_sum/len(draws),validation=val);history.append(record)
        with (RUN/'history.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        checkpoint('head-only-last.pt',epoch,val)
        if val['macro_top1']>best_score:
            best_score=val['macro_top1'];best_epoch=epoch;bad=0;checkpoint('head-only-best.pt',epoch,val)
        else:bad+=1
        print('EPOCH',json.dumps(record),flush=True)
        if bad>=3:break
    best=shared.load_checkpoint(RUN/'head-only-best.pt')
    head.load_state_dict({k.removeprefix('classifier.'):v for k,v in best['ema'].items() if k.startswith('classifier.')})
    after=shared.evaluate(head,features,eval_idx,evaluation.rows,labels,classes,set(old_index),grades,device,'after')
    assert all(torch.equal(v,best['ema'][k]) for k,v in initial.items() if not k.startswith('classifier.'))
    full=shared.build_model(base['config'],nclasses);full.load_state_dict(best['ema'],strict=True);full.to(device).eval()
    checks=[]
    with torch.inference_mode():
        for i in [0,len(candidates)-1,len(candidates),len(rows)-1,int(missing_indices[0])]:
            image,_=dataset[i]
            direct=full(image.unsqueeze(0).to(device));cached=head(torch.from_numpy(np.array(features[i:i+1],copy=True)).to(device))
            assert torch.equal(direct.topk(5).indices,cached.topk(5).indices)
            checks.append(dict(sample=rows[i]['sample_id'],max_logit_difference=float((direct-cached).abs().max())))
    result=dict(status='complete',classes=nclasses,best_epoch=best_epoch,epochs=len(history),before=before,after=after,
                initial_validation=initial_val,best_validation=best['validation'],frozen_backbone_unchanged=True,
                training_evaluation_hash_overlap=0,full_checkpoint_checks=checks,
                reused_features=len(rows)-len(missing_indices),new_features=len(missing_indices),
                checkpoint_sha256=shared.sha(RUN/'head-only-best.pt'),total_seconds=time.time()-started)
    (RUN/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('COMPLETE',json.dumps({'best_epoch':best_epoch,'minutes':result['total_seconds']/60}),flush=True)


if __name__=='__main__':
    main()
