"""Export the trained full checkpoint and verify mobile inference parity."""
import sys, json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/GlyphWeave'))
from handwriting.inference import Recognizer
from PIL import Image, ImageDraw
out=ROOT/'hanja_app/assets';out.mkdir(parents=True,exist_ok=True)
r=Recognizer(ROOT/'results/hanja3500-head-only-v1/head-only-best.pt',device='cpu')
x=torch.zeros(1,1,128,128)
torch.onnx.export(r.model,x,str(out/'hanja3500.onnx'),input_names=['image'],output_names=['logits'],opset_version=17,dynamo=False)
(out/'classes.json').write_text(json.dumps(r.classes,ensure_ascii=False))
s=ort.InferenceSession(str(out/'hanja3500.onnx'),providers=['CPUExecutionProvider'])
checks=[]
for name,strokes in [('一',[[(30,100),(220,100)]]),('十',[[(30,100),(220,100)],[(125,30),(125,220)]]),('大',[[(30,90),(220,90)],[(130,30),(115,140),(35,220)],[(125,120),(210,220)]])]:
 im=Image.new('L',(256,256),255);d=ImageDraw.Draw(im)
 for points in strokes:d.line(points,fill=0,width=7)
 tensor=r.transform(im).unsqueeze(0)
 with torch.no_grad(): a=r.model(tensor).numpy()[0]
 b=s.run(None,{'image':tensor.numpy()})[0][0]
 ids=np.argsort(-b)[:5];diff=float(np.max(np.abs(a-b)))
 assert diff<0.001 and np.array_equal(np.argsort(-a)[:5],ids)
 checks.append(dict(sample=name,top5=[r.classes[i] for i in ids],max_logit_difference=diff))
(out/'export_validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
print(json.dumps(checks,ensure_ascii=False,indent=2))
