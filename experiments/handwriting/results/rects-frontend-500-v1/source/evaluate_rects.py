"""Evaluate fixed restoration checkpoints on labeled characters from ReCTS photos."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from paths import APP_ASSETS, DATASETS, RESULTS
from .evaluate import FrozenHanjaEvaluator, _binary, normalize_ink
from .model import PhotoInkRestorer


def _crop_character(image, points):
    points = np.asarray(points, dtype=float).reshape(4, 2)
    left, top = points.min(axis=0)
    right, bottom = points.max(axis=0)
    if not np.isfinite(points).all() or right <= left or bottom <= top:
        raise ValueError('Invalid character box')
    margin = .05 * max(right - left, bottom - top)
    box = (max(0, int(np.floor(left - margin))), max(0, int(np.floor(top - margin))),
           min(image.width, int(np.ceil(right + margin))), min(image.height, int(np.ceil(bottom + margin))))
    crop = image.crop(box)
    if min(crop.size) < 1:
        raise ValueError('Empty character crop')
    rgb = np.asarray(crop)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    background = tuple(np.median(border, axis=0).astype(np.uint8).tolist())
    side = max(crop.size)
    square = Image.new('RGB', (side, side), background)
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return square.resize((256, 256), Image.Resampling.BICUBIC)


def _metrics(rows):
    return {method: {f'top{k}': sum(r['methods'][method]['rank'] <= k for r in rows) / len(rows)
                     for k in (1, 5, 10)} for method in ('auto_binary', 'previous', 'restored')} if rows else {}


def _save_preview(examples, path):
    font = ImageFont.truetype('/System/Library/Fonts/STHeiti Medium.ttc', 18)
    sheet = Image.new('RGB', (640, 190 * len(examples)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (photo, before, after, row) in enumerate(examples):
        for j, picture in enumerate([photo, Image.fromarray(before), Image.fromarray(after)]):
            sheet.paste(picture.convert('RGB').resize((160, 160)), (j * 160, i * 190))
        draw.text((3, i * 190 + 163), f"GT {row['character']} | old #{row['methods']['previous']['rank']} | new #{row['methods']['restored']['rank']}", font=font, fill='black')
        draw.text((485, i * 190 + 20), 'new top1:\n'+row['methods']['restored']['top10'][0], font=font, fill='black')
    sheet.save(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=DATASETS / 'rects-500')
    parser.add_argument('--output', type=Path, default=RESULTS / 'photo-restoration-rects-500-v1')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'result.json').exists():
        raise RuntimeError('Completed evaluation already exists')
    torch.set_num_threads(4)
    started = time.time()
    manifest = json.loads((args.data / 'manifest.json').read_text())
    classifier = FrozenHanjaEvaluator()
    labels = {c: i for i, c in enumerate(classifier.classes)}
    checkpoints = dict(previous=RESULTS / 'photo-restoration-wide-500-v1/epoch-100.pt',
                       restored=RESULTS / 'photo-restoration-half-thickness-100-v1/epoch-100.pt')
    models = {}
    for name, path in checkpoints.items():
        model = PhotoInkRestorer(48).to('mps')
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'])
        models[name] = model.eval()
    settings = dict(photos=len(manifest['selected_images']), source=manifest['source'], seed=manifest['seed'],
        split=manifest['split'], crop='annotated axis-aligned bounds +5% max-side margin, aspect-preserving square padding with median border color; 256px',
        scope='character recognition with ground-truth boxes, not full-photo detection',
        cutoff=.5, label_policy='exact Unicode match only; no simplified/traditional conversion',
        checkpoints={n:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in checkpoints.items()},
        classifier_sha256=hashlib.sha256((APP_ASSETS/'hanja.onnx').read_bytes()).hexdigest())
    (args.output/'settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2))
    micro = json.loads((RESULTS/'photo-restoration-micro-capacity-v1/split.json').read_text())
    trained_chars = {r['character'] for r in micro['records']['train']}
    counts, excluded = Counter(), Counter()
    tasks, photos = [], {}
    for filename in manifest['selected_images']:
        photo_id = Path(filename).stem
        annotation = json.loads((args.data/'gt_unicode'/f'{photo_id}.json').read_text())
        with Image.open(args.data/filename) as original:
            image = original.convert('RGB')
            accepted = 0
            for index, char in enumerate(annotation['chars']):
                counts['annotated_characters'] += 1
                text = char['transcription']
                if char.get('ignore',0):
                    counts['ignored'] += 1; continue
                if text not in labels:
                    counts['unsupported'] += 1; excluded[text] += 1; continue
                try: crop = _crop_character(image, char['points'])
                except ValueError:
                    counts['invalid_box'] += 1; continue
                tasks.append((crop, dict(photo_id=photo_id, index=index, character=text,
                    points=char['points'], seen_in_restoration_training=text in trained_chars)))
                accepted += 1
            photos[photo_id] = accepted
    counts['photos_with_supported_characters'] = sum(n>0 for n in photos.values())
    counts['evaluated_characters'] = len(tasks)
    print('SCOPE',json.dumps(dict(counts)),flush=True)
    rows, failures, successes = [], [], []
    with (args.output/'predictions.jsonl').open('w') as stream:
        for start in range(0,len(tasks),16):
            batch=tasks[start:start+16]
            inputs=np.stack([np.asarray(p,dtype=np.float32).transpose(2,0,1)/255 for p,_ in batch])
            tensor=torch.from_numpy(inputs).to('mps')
            masks={name:model(tensor).sigmoid().cpu().numpy()[:,0] for name,model in models.items()}
            for offset,(photo,row) in enumerate(batch):
                gray=np.asarray(photo.convert('L'))
                images={'auto_binary':_binary(gray,True)}
                for name in models:
                    ink=np.where(masks[name][offset]>=.5,masks[name][offset],0)
                    images[name]=np.uint8(np.clip(1-ink,0,1)*255)
                row['methods']={name:classifier.recognize(normalize_ink(im),labels[row['character']]) for name,im in images.items()}
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');rows.append(row)
                examples=successes if row['methods']['restored']['rank']==1 else failures
                if len(examples)<12:examples.append((photo,images['previous'],images['restored'],row))
            if start%160==0:
                progress=dict(state='evaluating',completed=len(rows),total=len(tasks),seconds=time.time()-started)
                (args.output/'progress.json').write_text(json.dumps(progress))
                print('PROGRESS',json.dumps(progress),flush=True)
    groups={c:[r for r in rows if r['character']==c] for c in sorted({r['character'] for r in rows})}
    result=dict(settings=settings,counts=dict(counts),unique_characters=len(groups),metrics=_metrics(rows),
        seen_characters=_metrics([r for r in rows if r['seen_in_restoration_training']]),
        unseen_characters=_metrics([r for r in rows if not r['seen_in_restoration_training']]),
        seen_count=sum(r['seen_in_restoration_training'] for r in rows),
        macro_top1={m:float(np.mean([_metrics(g)[m]['top1'] for g in groups.values()])) for m in models},
        unsupported_characters=dict(excluded.most_common()),seconds=time.time()-started)
    (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    _save_preview(failures,args.output/'failures.png');_save_preview(successes,args.output/'successes.png')
    (args.output/'progress.json').write_text(json.dumps(dict(state='complete',completed=len(rows))))
    print('COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ('settings','unsupported_characters')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
