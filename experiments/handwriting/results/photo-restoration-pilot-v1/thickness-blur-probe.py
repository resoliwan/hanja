from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageFilter,ImageDraw
import torch
from photo_restoration.model import PhotoInkRestorer
p=Path('experiments/handwriting/results/photo-restoration-pilot-v1');torch.set_num_threads(2)
a=np.load(p/'validation.npy',mmap_mode='r');indices=np.random.default_rng(74).choice(len(a),32,replace=False)
m=PhotoInkRestorer().to('mps').eval();m.load_state_dict(torch.load(p/'best.pt',map_location='cpu',weights_only=False)['model'])
results=[];examples=[]
for dilation in [0,4,10,20]:
 for blur in [0,1,3]:
  photos=[];truths=[];interiors=[]
  for i in indices:
   mask=Image.fromarray(255-a[i]).resize((256,256),Image.Resampling.BICUBIC)
   if dilation: mask=mask.filter(ImageFilter.MaxFilter(dilation+1))
   alpha=np.asarray(mask,dtype=np.float32)/255
   image=Image.fromarray(np.uint8(55+(245-55)*alpha)).convert('RGB').filter(ImageFilter.GaussianBlur(blur))
   photos.append(np.asarray(image,dtype=np.float32).transpose(2,0,1).copy()/255)
   truth=alpha>.9;truths.append(truth)
   interiors.append(np.asarray(Image.fromarray(np.uint8(truth)*255).filter(ImageFilter.MinFilter(7)))>0)
  predictions=[]
  with torch.inference_mode():
   for j in range(0,len(photos),16):predictions.extend(m(torch.from_numpy(np.stack(photos[j:j+16])).to('mps')).sigmoid().cpu().numpy()[:,0])
  pred=np.stack(predictions);truth=np.stack(truths);inside=np.stack(interiors)
  results.append(dict(dilation_pixels=dilation,blur_sigma=blur,n=32,ink_recall=float((pred[truth]>=.5).mean()),interior_recall=float((pred[inside]>=.5).mean()),interior_probability=float(pred[inside].mean())))
  if blur in [0,3]: examples.append((photos[0],pred[0],f'width +{dilation}px, blur {blur}'))
sheet=Image.new('RGB',(512,278*len(examples)),'white');draw=ImageDraw.Draw(sheet)
for i,(photo,pred,label) in enumerate(examples):
 sheet.paste(Image.fromarray(np.uint8(photo.transpose(1,2,0)*255)),(0,i*278));sheet.paste(Image.fromarray(np.uint8((1-pred)*255)).convert('RGB'),(256,i*278));draw.text((3,i*278+259),label,fill='black')
sheet.save(p/'thickness-blur-probe.png')
(p/'thickness-blur-probe.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
