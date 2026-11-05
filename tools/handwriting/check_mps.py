from paths import DATASETS, MODELS, RESULTS
import sys,json,time
sys.path.insert(0,'scripts')
from benchmark import render
from handwriting.inference import Recognizer
import torch
torch.set_num_threads(4)
print('MPS available',torch.backends.mps.is_available(),flush=True)
r=Recognizer(str(MODELS / 'glyphweave-v1.0.0-inference/best.pt'),'mps')
s=json.load(open(DATASETS / 'tradchar-first100.json'))
old=[x for x in json.load(open(RESULTS / 'tradchar-first100/predictions.json')) if x['model']=='GlyphWeave']
cs=[];t=time.perf_counter()
with torch.inference_mode():
 for i in range(0,100,32):
  x=torch.stack([r.transform(render(v['strokes'])) for v in s[i:i+32]]).to(r.device)
  ids=r.model(x).float().softmax(-1).topk(5,dim=-1).indices.cpu().tolist()
  cs.extend([[r.classes[j] for j in a] for a in ids])
print('elapsed',time.perf_counter()-t,'candidate mismatches',sum(a!=b['candidates'] for a,b in zip(cs,old)),flush=True)
