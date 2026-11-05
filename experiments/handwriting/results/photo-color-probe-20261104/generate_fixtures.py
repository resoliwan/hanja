from pathlib import Path
import json
import numpy as np
from PIL import Image
root=Path('/Users/leo/aoo_workspace/hanja/experiments/handwriting/results/photo-color-probe-20261104')
(root/'fixtures').mkdir(parents=True,exist_ok=True)
manifest=[]
for name,char in [('cho','超'),('yeok','譯')]:
    original=Image.open(root/'sources'/f'{name}.png').convert('RGB')
    a=np.asarray(original.convert('L'),dtype=float)
    # Fixed soft foreground mask derived once from the source photo; all variants share the same shape.
    alpha=np.clip((a-120)/100,0,1)[...,None]
    h,w=a.shape
    x=np.linspace(0,1,w)[None,:,None]
    ramp=np.where(x<.5, np.array([255,0,0])*(1-2*x)+np.array([0,255,0])*(2*x),np.array([0,255,0])*(2-2*x)+np.array([0,0,255])*(2*x-1))
    for bg_name,bg in [('light',245),('dark',55)]:
        colors={'neutral':np.array([15,15,15]) if bg_name=='light' else np.array([245,245,245]),'red':np.array([255,0,0]),'blue':np.array([0,0,255]),'green':np.array([0,255,0]),'rgb-gradient':ramp}
        for color,fg in colors.items():
            pixels=bg*(1-alpha)+fg*alpha
            case=f'{bg_name}-{color}'
            path=root/'fixtures'/f'{name}-{case}.png'
            Image.fromarray(np.uint8(np.clip(pixels,0,255))).save(path)
            manifest.append(dict(id=f'{name}-{case}',character=char,case=case,path=str(path)))
        for color in ['red','blue','green']:
            pixels=bg*(1-alpha)+colors[color]*alpha
            # White illumination across the middle affects both foreground and background.
            glare=.8*np.exp(-((x-.5)/.18)**2)
            pixels=pixels*(1-glare)+255*glare
            case=f'{bg_name}-{color}-center-glare'
            path=root/'fixtures'/f'{name}-{case}.png'
            Image.fromarray(np.uint8(np.clip(pixels,0,255))).save(path)
            manifest.append(dict(id=f'{name}-{case}',character=char,case=case,path=str(path)))
    path=root/'fixtures'/f'{name}-original.png'
    original.save(path)
    manifest.append(dict(id=f'{name}-original',character=char,case='original',path=str(path)))
(root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(root, len(manifest))
