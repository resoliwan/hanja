"""Photo-level ReCTS splits and cached character crops for frontend learning."""
import hashlib
import json
import random
from collections import Counter

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

from paths import DATASETS
from .evaluate_rects import _crop_character
from .continue_capacity import _atomic_json

DATA = DATASETS / 'rects-training-500'
HELD_OUT = DATASETS / 'rects-500'
SEED = 20261106


def _pixel_hash(path):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def prepare(output, classes):
    training = json.loads((DATA / 'manifest.json').read_text())
    held_out = json.loads((HELD_OUT / 'manifest.json').read_text())
    assert not set(training['selected_images']) & set(held_out['selected_images'])
    held_hashes = {_pixel_hash(HELD_OUT / name) for name in held_out['selected_images']}
    seen = set(held_hashes)
    candidates, duplicates = [], []
    for name in training['selected_images']:
        digest = _pixel_hash(DATA / name)
        if digest in seen:
            duplicates.append(name)
        else:
            seen.add(digest)
            candidates.append(dict(path=name, pixel_sha256=digest))
    random.Random(SEED).shuffle(candidates)
    validation_count = round(len(candidates) * .2)
    splits = dict(validation=candidates[:validation_count], train=candidates[validation_count:])
    assert not {r['pixel_sha256'] for r in splits['train']} & {r['pixel_sha256'] for r in splits['validation']}
    labels = {c: i for i,c in enumerate(classes)}
    records, summaries = {}, {}
    for split, photos in splits.items():
        rows, excluded = [], Counter()
        for photo in photos:
            name = photo['path']
            annotation = json.loads((DATA / 'gt_unicode' / (name.split('/')[-1].replace('.jpg','.json'))).read_text())
            for index, char in enumerate(annotation['chars']):
                text = char['transcription']
                if char.get('ignore',0):
                    excluded['ignored'] += 1
                elif text not in labels:
                    excluded['unsupported'] += 1
                else:
                    rows.append(dict(photo=name,index=index,character=text,label=labels[text],points=char['points']))
        assert rows, 'No supported characters in split'
        cache_path=output/f'{split}-rgb.npy'
        cache=np.lib.format.open_memmap(cache_path,mode='w+',dtype=np.uint8,shape=(len(rows),256,256,3))
        last_path=None
        for index,row in enumerate(rows):
            if row['photo'] != last_path:
                with Image.open(DATA / row['photo']) as source:image=source.convert('RGB')
                last_path=row['photo']
            cache[index]=np.asarray(_crop_character(image,row['points']))
        cache.flush();del cache
        records[split]=rows
        summaries[split]=dict(photos=len(photos),characters=len(rows),unique_characters=len({r['character'] for r in rows}),excluded=dict(excluded))
    result=dict(seed=SEED,photos=splits,records=records,summary=summaries,excluded_duplicate_photos=duplicates,
        held_out_photos=len(held_out['selected_images']),isolation='photo IDs and exact decoded RGB pixel hashes; near-duplicate scenes not guaranteed independent')
    _atomic_json(output/'split.json',result)
    return result


class ReCTSCharacters(Dataset):
    def __init__(self, cache_path, records):
        self.images=np.load(cache_path,mmap_mode='r')
        self.records=records

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        rgb=np.asarray(self.images[index],dtype=np.float32).transpose(2,0,1).copy()/255
        return torch.from_numpy(rgb),self.records[index]['label']
