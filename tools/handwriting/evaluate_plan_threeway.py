"""Evaluate unchanged public models on the plan's locked evaluation-only images."""
from paths import DATASETS, MODELS, RESULTS, ROOT
import argparse,csv,json,hashlib,io,tarfile,time,sys
from pathlib import Path
import numpy as np
from PIL import Image
BASE=DATASETS / 'korean-hanja3500-v1';OUT=RESULTS / 'plan-threeway-evaluation'

def prepare():
 rows=[];seen=set()
 for source in ['fonts','scut']:
  for r in csv.DictReader((BASE/'manifests'/f'{source}.csv').open()):
   if r['usage']!='evaluation_only' or r['pixel_sha256'] in seen:continue
   seen.add(r['pixel_sha256']);r['evaluation_set']='scut' if source=='scut' else 'font_original' if r['variant'].endswith(':base') else 'font_variants';rows.append(r)
 from collections import Counter
 counts=Counter(r['evaluation_set'] for r in rows)
 assert counts=={'font_original':3500,'font_variants':17500,'scut':52987},counts
 assert len({r['sample_id'] for r in rows})==73987
 classes=(BASE/'classes.txt').read_text().splitlines();assert len(set(classes))==3500
 assert all(classes[int(r['class_id'])]==r['character'] for r in rows)
 payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
 (OUT/'evaluation-manifest.jsonl').write_text(payload)
 provenance={'counts':dict(counts),'classes_sha256':hashlib.sha256((BASE/'classes.txt').read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(payload.encode()).hexdigest(),'target':'simpledic Daehan Geomjeonghoe cumulative 3500, per training plan','training_performed':False,'model_files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [MODELS / 'glyphweave-v1.0.0-inference/best.pt',MODELS / 'shikiji.onnx']}}
 (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2));print(provenance,flush=True)

def run(model):
 rows=[json.loads(l) for l in (OUT/'evaluation-manifest.jsonl').read_text().splitlines()]
 path=OUT/(model+'-scores.jsonl');old=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
 assert [r['id'] for r in old]==[r['sample_id'] for r in rows[:len(old)]]
 tars={}
 def load(row):
  shard=row['shard']
  if shard not in tars:tars[shard]=tarfile.open(BASE/shard)
  im=Image.open(io.BytesIO(tars[shard].extractfile(row['member']).read())).convert('L');im.load()
  assert im.size==(128,128) and hashlib.sha256(im.tobytes()).hexdigest()==row['pixel_sha256'],row['sample_id']
  return im
 if model=='glyph':
  import torch
  from handwriting.inference import Recognizer
  torch.set_num_threads(2);engine=Recognizer(str(MODELS / 'glyphweave-v1.0.0-inference/best.pt'),'mps');classes=engine.classes
  def infer(images):
   with torch.inference_mode():
    x=torch.stack([engine.transform(im) for im in images]).to(engine.device)
    probs=engine.model(x).float().softmax(-1);scores,ids=probs.topk(5,dim=-1)
    return ids.cpu().numpy(),scores.cpu().numpy()
 else:
  import onnxruntime as ort
  meta=json.loads((MODELS / 'shikiji.metadata.json').read_text());classes=meta['label_chars']
  opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
  engine=ort.InferenceSession(str(MODELS / 'shikiji.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
  def infer(images):
   x=np.stack([np.asarray(im.convert('RGB').resize((160,160),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1)/255.0 for im in images])
   logits=engine.run([meta['output_name']],{meta['input_name']:x})[0]
   exp=np.exp(logits-logits.max(axis=1,keepdims=True));probs=exp/exp.sum(axis=1,keepdims=True);ids=np.argsort(-logits,axis=1)[:,:5]
   return ids,np.take_along_axis(probs,ids,axis=1)
 start=time.perf_counter();print(model,'start',len(old),'/',len(rows),flush=True)
 with path.open('a',buffering=1) as f:
  for i in range(len(old),len(rows),32):
   batch=rows[i:i+32];images=[load(r) for r in batch];ids,scores=infer(images)
   assert np.isfinite(scores).all() and (scores>=0).all() and (scores<=1).all()
   for r,ii,vv in zip(batch,ids,scores):
    out={'id':r['sample_id'],'label':r['character'],'set':r['evaluation_set'],'grade':r['grade'],'candidates':[{'character':classes[int(j)],'score':float(v)} for j,v in zip(ii,vv)]}
    f.write(json.dumps(out,ensure_ascii=False)+'\n')
   if i//1000!=(i+len(batch))//1000 or i+len(batch)==len(rows):print(model,i+len(batch),'/',len(rows),'elapsed_s',round(time.perf_counter()-start,1),flush=True)
 for t in tars.values():t.close()
 print(model,'COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','glyph','shikiji']);a=p.parse_args()
 if a.action=='prepare':prepare()
 else:run(a.action)
