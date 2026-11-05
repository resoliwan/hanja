"""Select disjoint source groups and cache a balanced 500-character pilot."""

import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from hanja3500_dataset import Hanja3500Dataset
from paths import DATASETS, ROOT

sys.path.insert(0, str(ROOT / 'vendor/GlyphWeave'))
from handwriting.images import normalize_glyph


def prepare_pilot(output, seed=20261104):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'split.json').exists():
        return json.loads((output / 'split.json').read_text())
    data = DATASETS / 'korean-hanja5000-v1'
    candidates = Hanja3500Dataset(data, 'train')
    evaluation = Hanja3500Dataset(data, 'evaluation')
    valid_ids = set((data / 'validation-sample-ids.txt').read_text().splitlines())
    pools = {name: collections.defaultdict(list) for name in ('train', 'validation', 'test')}
    for i, row in enumerate(candidates.rows):
        split = 'validation' if row['sample_id'] in valid_ids else 'train'
        pools[split][int(row['class_id'])].append(i)
    for i, row in enumerate(evaluation.rows):
        pools['test'][int(row['class_id'])].append(i)
    rng = np.random.default_rng(seed)
    grades = collections.defaultdict(list)
    for class_id, indices in pools['train'].items():
        if len(indices) >= 40 and pools['validation'][class_id] and len(pools['test'][class_id]) >= 2:
            grades[candidates.rows[indices[0]]['grade']].append(class_id)
    # Cycle grades instead of letting advanced-level classes dominate sampling.
    for values in grades.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < 500:
        for grade in sorted(grades):
            if grades[grade] and len(selected) < 500:
                selected.append(grades[grade].pop())
    records = {}
    for split, count in [('train', 40), ('validation', 1), ('test', 2)]:
        dataset = evaluation if split == 'test' else candidates
        indices = []
        for class_id in selected:
            available = pools[split][class_id]
            # Prefer one held-out font and one handwriting example for final test.
            if split == 'test':
                fonts = [i for i in available if dataset.rows[i]['source'] == 'fonts']
                handwriting = [i for i in available if dataset.rows[i]['source'] == 'scut']
                chosen = [int(rng.choice(fonts)), int(rng.choice(handwriting))] if fonts and handwriting else rng.choice(available, count, replace=False).tolist()
            else:
                chosen = rng.choice(available, count, replace=False).tolist()
            indices.extend(chosen)
        records[split] = [dataset.rows[i] for i in indices]
        cache = np.lib.format.open_memmap(output / f'{split}.npy', mode='w+', dtype=np.uint8, shape=(len(indices), 128, 128))
        for j, i in enumerate(indices):
            image, _ = dataset[i]
            cache[j] = np.asarray(normalize_glyph(image, 128))
            if j % 4000 == 0:
                print(f'PREPARE {split} {j}/{len(indices)}', flush=True)
        cache.flush()
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        for key in ('pixel_sha256', 'group_id', 'sample_id'):
            overlap = {r[key] for r in records[a]} & {r[key] for r in records[b]}
            assert not overlap, (a, b, key, len(overlap))
    payload = dict(seed=seed, selected_classes=selected, records=records,
        source_manifest_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (data / 'manifests').glob('*.csv')})
    (output / 'split.json').write_text(json.dumps(payload, ensure_ascii=False))
    candidates.close()
    evaluation.close()
    print('PREPARE COMPLETE', {s: len(v) for s, v in records.items()}, flush=True)
    return payload
