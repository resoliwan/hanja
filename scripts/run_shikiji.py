"""Shikiji v0.0.2 ONNX evaluation on the same ordered SCUT TradChar set."""
import argparse,json,time,hashlib,statistics,csv
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image
from benchmark import ROOT,render
OUT=ROOT/'results/shikiji-set1'
meta=json.loads((ROOT/'models/shikiji.metadata.json').read_text())
classes=meta['label_chars']
assert len(classes)==meta['num_classes']==10596
assert meta['preprocessing']['normalization']=='none'

def make_input(s):
    # Same stroke rendering as the previous benchmark, then documented RGB resize.
    im=render(s['strokes']).convert('RGB').resize((160,160),Image.Resampling.BILINEAR)
    return np.asarray(im,dtype=np.float32).transpose(2,0,1)/255.0

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    samples=json.loads((ROOT/'data/tradchar-set1.json').read_text())
    opts=ort.SessionOptions();opts.intra_op_num_threads=8;opts.inter_op_num_threads=1
    sess=ort.InferenceSession(str(ROOT/'models/shikiji.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
    print('input',[(v.name,v.shape,v.type) for v in sess.get_inputs()],'output',[(v.name,v.shape) for v in sess.get_outputs()],flush=True)
    if a.smoke:samples=samples[:100]+[s for s in samples if s['label']=='學']
    path=OUT/('smoke.jsonl' if a.smoke else 'predictions.jsonl')
    old=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    assert [r['id'] for r in old]==[s['id'] for s in samples[:len(old)]]
    with path.open('a',buffering=1) as f:
        for i in range(len(old),len(samples),32):
            ss=samples[i:i+32];t=time.perf_counter()
            try:
                x=np.stack([make_input(s) for s in ss]);logits=sess.run([meta['output_name']],{meta['input_name']:x})[0]
                assert logits.shape==(len(ss),len(classes)) and np.isfinite(logits).all()
                ids=np.argsort(-logits,axis=1)[:,:5];pred=[[classes[int(j)] for j in idx] for idx in ids];error=None
            except Exception as e:
                raise RuntimeError(f'Batch {i} failed; no samples silently excluded') from e
            ms=(time.perf_counter()-t)*1000/len(ss)
            for s,c in zip(ss,pred):f.write(json.dumps(dict(id=s['id'],label=s['label'],candidates=c,top1=c[0]==s['label'],top5=s['label'] in c,error=error,covered=s['label'] in classes,ms=ms),ensure_ascii=False)+'\n')
            if (i//1000!=(i+len(ss))//1000) or a.smoke:print(i+len(ss),'/',len(samples),flush=True)
    rows=[json.loads(l) for l in path.read_text().splitlines()]
    summary=dict(n=len(rows),top1=sum(r['top1'] for r in rows),top5=sum(r['top5'] for r in rows),errors=sum(bool(r['error']) for r in rows),covered_samples=sum(r['covered'] for r in rows),unique_labels=len(set(s['label'] for s in samples)),missing_labels=sorted(set(s['label'] for s in samples)-set(classes)),median_amortized_ms=statistics.median(r['ms'] for r in rows),providers=sess.get_providers(),model_sha256=hashlib.sha256((ROOT/'models/shikiji.onnx').read_bytes()).hexdigest())
    (OUT/('smoke-summary.json' if a.smoke else 'summary.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print({k:v for k,v in summary.items() if k!='missing_labels'},flush=True)
    print('學 supported', '學' in classes,'学 supported','学' in classes,flush=True)
    for r in rows:
        if r['label']=='學':print(r,flush=True)
    if not a.smoke:
        with (OUT/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(dict(r,candidates=' '.join(r['candidates'])) for r in rows)
if __name__=='__main__':main()
