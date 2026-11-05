from pathlib import Path
import json,hashlib
import numpy as np
import onnxruntime as ort
from PIL import Image,ImageDraw,ImageFont
root=Path('/Users/leo/aoo_workspace/hanja')
out=root/'experiments/handwriting/results/photo-color-probe-20261104'
manifest=json.loads((out/'manifest.json').read_text())
classes=json.loads((root/'apps/dic/assets/classes.json').read_text())
s=ort.InferenceSession(str(root/'apps/dic/assets/hanja.onnx'),providers=['CPUExecutionProvider'])
methods=['direct-gray','gray-normalized','current-binary','auto-polarity-binary']
results=[]
for row in manifest:
    for method in methods:
        tensor=np.fromfile(out/'tensors'/f'{row["id"]}-{method}.f32',dtype=np.float32).reshape(1,1,128,128)
        scores=s.run(None,{'image':tensor})[0].flatten()
        order=np.argsort(-scores)
        results.append({**row,'method':method,'rank':int(np.where(order==classes.index(row['character']))[0][0])+1,'top10':[classes[i] for i in order[:10]]})
(out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
summary={m:{'top1':sum(r['rank']==1 for r in results if r['method']==m and r['case']!='original'),'top10':sum(r['rank']<=10 for r in results if r['method']==m and r['case']!='original'),'total':32} for m in methods}
print(json.dumps(summary,indent=2))
cases=list(dict.fromkeys(r['case'] for r in manifest))
lines=['# Photo color recognition probe','', 'Two glyphs (超, 譯) from IMG_4431.HEIC; synthetic variants, not a population accuracy benchmark.','', 'Fixed soft glyph mask: clip((source luminance - 120) / 100, 0, 1). Uniform backgrounds: light=245, dark=55. Primaries: RGB 255/0. RGB gradient: red→green→blue horizontally. Center glare: 80% white blending at center, Gaussian width 0.18 × crop width, applied to foreground and background. This reconstruction simplifies original texture; original crops are separate controls.','', 'Model: apps/dic/assets/hanja.onnx, SHA256 '+hashlib.sha256((root/'apps/dic/assets/hanja.onnx').read_bytes()).hexdigest(),'', 'All tensors generated using Flutter image package and actual app normalizeInk/normalizePhotoInk functions. Direct-gray skips bounding-box normalization; gray-normalized uses existing handwriting normalization. Auto-polarity uses Otsu + majority of crop perimeter to decide which intensity class is background; it is experimental and can fail with gradients or mixed contrast.','', '|Case|Direct gray|Gray + existing normalization|Current binary|Auto polarity + binary|','|---|---|---|---|---|']
for case in cases:
    cells=[]
    for m in methods:
        rows=[r for r in results if r['case']==case and r['method']==m]
        cells.append('/'.join(str(r['rank']) for r in rows))
    lines.append('|'+case+'|'+'|'.join(cells)+'|')
    print(case,*cells)
lines+=['','Each cell gives expected-class ranks in order 超 / 譯.','', '## Synthetic variant totals','',json.dumps(summary,indent=2),'','Reproduce: generate_fixtures.py; copy export_tensors_test.dart to apps/dic/test/photo_color_probe_tmp_test.dart, run flutter test from apps/dic, then evaluate.py. Remove temporary test after run. Scripts use absolute workspace paths.']
(out/'REPORT.md').write_text('\n'.join(lines))
font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',14)
# Contact sheet of all fixtures, labeled with experimental auto-binary rank.
sheet=Image.new('RGB',(8*180,4*215),'white');draw=ImageDraw.Draw(sheet)
for i,row in enumerate(r for r in manifest if r['case']!='original'):
    x=(i%8)*180;y=(i//8)*215
    im=Image.open(row['path']);im.thumbnail((172,172));sheet.paste(im,(x+4,y))
    rank=next(r['rank'] for r in results if r['id']==row['id'] and r['method']=='auto-polarity-binary')
    label=row['case'].replace('center-glare','glare')
    draw.text((x+4,y+174),label,fill='black',font=font)
    draw.text((x+4,y+192),f'{row["id"].split("-")[0]} auto rank {rank}',fill='black',font=font)
sheet.save(out/'contact-sheet.png')
