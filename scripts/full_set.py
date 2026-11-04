"""Full-set evaluation, checkpointed per sample; preserves original preprocessing."""
import argparse,json,struct,zipfile,time,subprocess,statistics,csv,sys
from pathlib import Path
from benchmark import ROOT,normalize,render,Zinnia
OUT=ROOT/'results/tradchar-set1'
DATA=ROOT/'data/tradchar-set1.json'
def extract():
    with zipfile.ZipFile(ROOT/'data/onHCCTestDB-TradChar.zip') as z:
        b='onHCCTestDB-TradChar/'
        pot=z.read(b+'pot/onHCCTestDB-TradChar 1.pot')
        labels=z.read(b+'label/onHCCTestDB-TradChar 1.txt').decode('utf-8-sig').splitlines()
    samples=[];offset=0
    for i,label in enumerate(labels):
        size,code,nstrokes=struct.unpack_from('<HIH',pot,offset)
        assert code.to_bytes(4,'little')[:2][::-1].decode('gbk')==label
        pos=offset+8;strokes=[];stroke=[]
        while True:
            x,y=struct.unpack_from('<hh',pot,pos);pos+=4
            if (x,y)==(-1,-1):break
            if (x,y)==(-1,0):
                assert stroke;strokes.append(stroke);stroke=[]
            else:
                assert x>=0 and y>=0;stroke.append([x,y])
        assert pos==offset+size and len(strokes)==nstrokes and not stroke
        samples.append(dict(id=f'trad1-{i+1:04d}',label=label,strokes=strokes,record=i+1,byte_offset=offset))
        offset+=size
    assert offset==len(pot) and len(samples)==20000
    DATA.write_text(json.dumps(samples,ensure_ascii=False))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'manifest.json').write_text(json.dumps(dict(dataset='SCUT-onHCCTestDB TradChar subset 1, not COUCH2009',samples=len(samples),unique_labels=len(set(labels)),verified_labels=len(labels),pot_bytes=len(pot),selection='all records in original order, no exclusions'),indent=2))
    print('Extracted',len(samples),'samples;',len(set(labels)),'distinct characters')
def row(s,c,ms,error=None):
    return dict(id=s['id'],label=s['label'],candidates=c,ms=ms,error=error,top1=bool(c) and c[0]==s['label'],top5=s['label'] in c)
def run(model):
    OUT.mkdir(parents=True,exist_ok=True)
    samples=json.loads(DATA.read_text());path=OUT/(model+'.jsonl')
    existing=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    assert [r['id'] for r in existing]==[s['id'] for s in samples[:len(existing)]]
    start=len(existing);print(model,'starting',start,flush=True)
    engine=None;vocab=None
    if model=='zinnia':engine=Zinnia();vocab=engine.classes
    elif model=='cnn':
        from cnn_chinese_hw.recognizer.TFLiteRecognizer import TFLiteRecognizer
        engine=TFLiteRecognizer();vocab={chr(int(x)) for x in engine.classnames}
    elif model=='glyph':
        import torch
        from handwriting.inference import Recognizer
        torch.set_num_threads(4)
        engine=Recognizer(str(ROOT/'models/glyphweave-v1.0.0-inference/best.pt'),'mps');vocab=set(engine.classes)
    if vocab is not None:(OUT/(model+'-coverage.json')).write_text(json.dumps(dict(covered_samples=sum(s['label'] in vocab for s in samples),vocabulary_size=len(vocab),missing_labels=sorted(set(s['label'] for s in samples)-vocab)),ensure_ascii=False,indent=2))
    batch=32 if model=='glyph' else 100 if model=='hanzi' else 1
    with path.open('a',buffering=1) as f:
        for ix in range(start,len(samples),batch):
            ss=samples[ix:ix+batch];t=time.perf_counter()
            if model=='hanzi':
                temp=OUT/'hanzi-input.json';temp.write_text(json.dumps([dict(s,strokes=normalize(s['strokes'])) for s in ss],ensure_ascii=False))
                rs=json.loads(subprocess.check_output(['node',str(ROOT/'scripts/hanzi_lookup.cjs'),str(temp)]))
                assert len(rs)==len(ss)
                rr=[row(s,r['candidates'],r['ms'],r['error']) for s,r in zip(ss,rs)]
            elif model=='glyph':
                # Identical official transform and all-class top-5, batched for throughput.
                try:
                    with torch.inference_mode():
                        x=torch.stack([engine.transform(render(s['strokes'])) for s in ss]).to(engine.device)
                        probs=engine.model(x).float().softmax(-1)
                        idx=probs.topk(5,dim=-1).indices.cpu().tolist()
                    ms=(time.perf_counter()-t)*1000/len(ss)
                    rr=[row(s,[engine.classes[j] for j in ids],ms) for s,ids in zip(ss,idx)]
                except Exception as e:
                    # Preserve per-sample accounting if a malformed image breaks a batch.
                    rr=[]
                    for s in ss:
                        t1=time.perf_counter()
                        try:c=[v['character'] for v in engine.predict(render(s['strokes']),topk=5)['candidates']];error=None
                        except Exception as e:c=[];error=repr(e)
                        rr.append(row(s,c,(time.perf_counter()-t1)*1000,error))
            else:
                s=ss[0]
                try:
                    c=engine.predict(s['strokes']) if model=='zinnia' else [chr(int(v[1])) for v in engine.get_candidates_list(s['strokes'],n_cands=5)]
                    error=None
                except Exception as e:c=[];error=repr(e)
                rr=[row(s,c,(time.perf_counter()-t)*1000,error)]
            for r in rr:f.write(json.dumps(r,ensure_ascii=False)+'\n')
            if ix//1000!=(ix+len(ss))//1000 or ix+len(ss)==len(samples):print(model,ix+len(ss),'/',len(samples),flush=True)
    print(model,'COMPLETE',flush=True)
def report():
    summary={};allrows=[]
    for m in ['hanzi','zinnia','cnn','glyph']:
        rs=[json.loads(l) for l in (OUT/(m+'.jsonl')).read_text().splitlines()]
        assert len(rs)==20000 and len(set(r['id'] for r in rs))==20000
        summary[m]=dict(n=len(rs),top1=sum(r['top1'] for r in rs),top5=sum(r['top5'] for r in rs),errors=sum(bool(r['error']) for r in rs),median_ms=statistics.median(r['ms'] for r in rs))
        allrows.extend(dict(model=m,**r) for r in rs)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    with (OUT/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=allrows[0].keys());w.writeheader();w.writerows(dict(r,candidates=' '.join(r['candidates'])) for r in allrows)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['extract','hanzi','zinnia','cnn','glyph','report']);a=p.parse_args()
    if a.action=='extract':extract()
    elif a.action=='report':report()
    else:run(a.action)
