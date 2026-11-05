"""Evaluate unmodified GlyphWeave weights and preprocessing on the ReCTS subset."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from paths import ROOT, MODELS, RESULTS, DATASETS
from .evaluate import _binary, normalize_ink
from .evaluate_rects import _crop_character

sys.path.insert(0, str(ROOT / 'vendor/GlyphWeave'))
from handwriting.inference import Recognizer


def _summary(rows):
    result = {'n': len(rows), 'supported': sum(r['supported'] for r in rows)}
    for mode in ('original_preprocessing', 'same_binary_preprocessing'):
        result[mode] = {f'top{k}': sum(r[mode]['rank'] is not None and r[mode]['rank'] <= k for r in rows) / len(rows) if rows else None for k in (1, 5, 10)}
        result[mode]['rejected'] = sum('error' in r[mode] for r in rows)
    return result


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    output = RESULTS / 'glyphweave-original-rects-500-v1'
    output.mkdir(exist_ok=True)
    if (output / 'result.json').exists():
        raise RuntimeError('Evaluation already complete')
    checkpoint = MODELS / 'glyphweave-v1.0.0-inference/best.pt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == json.loads((checkpoint.parent / 'manifest.json').read_text())['checkpoint_sha256']
    recognizer = Recognizer(checkpoint, device='mps')
    labels = {c: i for i, c in enumerate(recognizer.classes)}
    data = DATASETS / 'rects-500'
    selection = json.loads((data / 'manifest.json').read_text())
    prior = [json.loads(line) for line in (RESULTS / 'photo-restoration-rects-500-v1/predictions.jsonl').read_text().splitlines()]
    old_ids = {(r['photo_id'], r['index']) for r in prior}
    tasks = []
    for filename in selection['selected_images']:
        photo_id = Path(filename).stem
        annotation = json.loads((data / 'gt_unicode' / f'{photo_id}.json').read_text())
        with Image.open(data / filename) as image:
            image = image.convert('RGB')
            for index, char in enumerate(annotation['chars']):
                text = char['transcription']
                is_han = len(text) == 1 and (0x3400 <= ord(text) <= 0x9fff or 0xf900 <= ord(text) <= 0xfaff or 0x20000 <= ord(text) <= 0x3134f)
                if char.get('ignore', 0) or not is_han:
                    continue
                photo = _crop_character(image, char['points'])
                row = dict(photo_id=photo_id, index=index, character=text, supported=text in labels, previous_cohort=(photo_id,index) in old_ids)
                tasks.append((photo, row))
    started = time.time()
    rows = []
    with (output / 'predictions.jsonl').open('w') as stream:
        for start in range(0, len(tasks), 32):
            batch = tasks[start:start + 32]
            for mode in ('original_preprocessing', 'same_binary_preprocessing'):
                tensors, indices = [], []
                for index, (photo, row) in enumerate(batch):
                    try:
                        if mode == 'original_preprocessing':
                            tensor = recognizer.transform(photo)
                        else:
                            gray = np.asarray(photo.convert('L'))
                            tensor = torch.from_numpy(normalize_ink(_binary(gray, True))[None])
                        tensors.append(tensor); indices.append(index)
                    except ValueError as error:
                        row[mode] = dict(rank=None, error=str(error), top10=[])
                if tensors:
                    scores = recognizer.model(torch.stack(tensors).to(recognizer.device)).float().cpu().numpy()
                    for index, score in zip(indices, scores):
                        row = batch[index][1]
                        order = np.argsort(-score)
                        rank = int(np.flatnonzero(order == labels[row['character']])[0])+1 if row['supported'] else None
                        row[mode] = dict(rank=rank, top10=[recognizer.classes[i] for i in order[:10]])
            for _, row in batch:
                rows.append(row); stream.write(json.dumps(row, ensure_ascii=False)+'\n')
            if start % 320 == 0:
                progress = dict(completed=len(rows), total=len(tasks), seconds=time.time()-started)
                (output/'progress.json').write_text(json.dumps(progress))
                print('PROGRESS', progress, flush=True)
    previous = [r for r in rows if r['previous_cohort']]
    assert len(previous) == len(prior) == 4449
    common = [r for r in previous if r['supported']]
    common_ids = {(r['photo_id'],r['index']) for r in common}
    prior_common = [r for r in prior if (r['photo_id'],r['index']) in common_ids]
    result = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest, output_classes=len(labels),
        photos=len(selection['selected_images']), scope='ground truth character crops; exact Unicode labels, no script conversion, full 7247-way ranking',
        all_han=_summary(rows), supported_han=_summary([r for r in rows if r['supported']]),
        same_4449=_summary(previous), common_vocabulary=_summary(common),
        current_classifier_on_common={f'top{k}':sum(r['methods']['auto_binary']['rank']<=k for r in prior_common)/len(prior_common) for k in (1,5,10)},
        seconds=time.time()-started, no_training=True)
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (output/'progress.json').write_text(json.dumps(dict(state='complete',completed=len(rows))))
    print('COMPLETE',json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
