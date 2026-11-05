"""Collect native full-vocabulary softmax top-5; no score calibration or renormalization."""
from paths import DATASETS, MODELS, RESULTS
import json,sys,argparse,unicodedata,time
from pathlib import Path
import numpy as np
from benchmark import render
OUT=RESULTS / 'ensemble-korean3500'
def main():
 p=argparse.ArgumentParser();p.add_argument('model',choices=['glyph','shikiji']);args=p.parse_args();model=args.model
 inventory=set((RESULTS / 'korean3500/inventory-3500.txt').read_text().strip())
 samples=[s for s in json.loads((DATASETS / 'tradchar-set1.json').read_text()) if unicodedata.normalize('NFC',s['label']) in inventory]
 assert len(samples)==11999
 if model=='glyph':
  import torch
  from handwriting.inference import Recognizer
  torch.set_num_threads(4);engine=Recognizer(str(MODELS / 'glyphweave-v1.0.0-inference/best.pt'),'mps');classes=engine.classes
  def infer(ss):
   with torch.inference_mode():
    x=torch.stack([engine.transform(render(s['strokes'])) for s in ss]).to(engine.device)
    prob=engine.model(x).float().softmax(-1)
    scores,ids=prob.topk(5,dim=-1)
    return ids.cpu().numpy(),scores.cpu().numpy()
 else:
  import onnxruntime as ort
  from run_shikiji import make_input,meta
  opts=ort.SessionOptions();opts.intra_op_num_threads=8;opts.inter_op_num_threads=1
  engine=ort.InferenceSession(str(MODELS / 'shikiji.onnx'),sess_options=opts,providers=['CPUExecutionProvider']);classes=meta['label_chars']
  def infer(ss):
   logits=engine.run([meta['output_name']],{meta['input_name']:np.stack([make_input(s) for s in ss])})[0]
   exp=np.exp(logits-logits.max(axis=1,keepdims=True));prob=exp/exp.sum(axis=1,keepdims=True)
   ids=np.argsort(-logits,axis=1)[:,:5]
   return ids,np.take_along_axis(prob,ids,axis=1)
 path=OUT/(model+'-scores.jsonl');old=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
 assert [r['id'] for r in old]==[s['id'] for s in samples[:len(old)]]
 with path.open('a',buffering=1) as f:
  for i in range(len(old),len(samples),32):
   ss=samples[i:i+32];ids,scores=infer(ss)
   assert np.isfinite(scores).all() and (scores>=0).all() and (scores<=1).all()
   for s,ii,vv in zip(ss,ids,scores):
    f.write(json.dumps(dict(id=s['id'],label=s['label'],candidates=[dict(character=classes[int(j)],score=float(v)) for j,v in zip(ii,vv)]),ensure_ascii=False)+'\n')
   if i//1000!=(i+len(ss))//1000 or i+len(ss)==len(samples):print(model,i+len(ss),'/',len(samples),flush=True)
if __name__=='__main__':main()
