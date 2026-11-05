from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from photo_restoration.model import PhotoInkRestorer
p=Path('experiments/handwriting/results/photo-restoration-pilot-v1'); source=p.parent/'photo-color-probe-20261104'
manifest={r['id']:r for r in json.loads((source/'manifest.json').read_text())};results={r['id']:r for r in json.loads((p/'known-photo-results.json').read_text())['rows']}
ids=['yeok-light-red','cho-light-green','yeok-dark-red-center-glare']
torch.set_num_threads(2);model=PhotoInkRestorer().to('mps').eval();model.load_state_dict(torch.load(p/'best.pt',map_location='cpu',weights_only=False)['model'])
font=ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc',22)
small=ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc',18)
sheet=Image.new('RGB',(600,1030),'#f1f3f5');draw=ImageDraw.Draw(sheet)
draw.text((55,12),'입력 사진',font=font,fill='black');draw.text((355,12),'복원 이미지',font=font,fill='black')
info=[]
for i,key in enumerate(ids):
 photo=Image.open(manifest[key]['path']).convert('RGB').resize((256,256),Image.Resampling.BICUBIC)
 data=np.asarray(photo,dtype=np.float32).transpose(2,0,1).copy()/255
 with torch.inference_mode():mask=model(torch.from_numpy(data[None]).to('mps')).sigmoid().cpu().numpy()[0,0]
 restored=Image.fromarray(np.uint8(np.clip(1-mask,0,1)*255)).convert('RGB')
 y=52+i*326;sheet.paste(photo,(18,y));sheet.paste(restored,(318,y))
 r=results[key];prediction=r['methods']['restored'];text=f"{i+1}. 정답 {r['character']} · 정답 순위 {prediction['rank']}위"
 draw.text((18,y+264),text,font=font,fill='black')
 description={'yeok-light-red':'밝은 배경 · 빨간 글씨','cho-light-green':'밝은 배경 · 초록 글씨','yeok-dark-red-center-glare':'어두운 배경 · 빨간 글씨 · 가운데 밝은 조명'}[key]
 draw.text((18,y+293),description,font=small,fill='#555555');info.append(dict(id=key,expected=r['character'],prediction=prediction['top10'][0],rank=prediction['rank']))
sheet.save(p/'three-failed-restoration-pairs.png');(p/'three-failed-restoration-pairs.json').write_text(json.dumps(info,ensure_ascii=False,indent=2));print(json.dumps(info,ensure_ascii=False))
