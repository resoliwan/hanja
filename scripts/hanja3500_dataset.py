"""Image/label loader for this dataset. No torch dependency until user transforms.

Training excludes SCUT, exact evaluation overlaps and conflicting pixel labels.
Synthetic augmentations share a group_id; keep groups together in any future split.
"""
import csv,io,tarfile
from pathlib import Path
from PIL import Image

class Hanja3500Dataset:
    def __init__(self, root, usage='train', sources=None, transform=None, deduplicate=True):
        self.root=Path(root);self.transform=transform;self.rows=[];self._tars={}
        self.classes=(self.root/'classes.txt').read_text().splitlines()
        if usage not in ('train','evaluation'):raise ValueError('usage must be train or evaluation')
        allowed=set(sources or (['fonts','ai-free'] if usage=='train' else ['fonts','scut']))
        if usage=='train' and 'scut' in allowed:raise ValueError('SCUT is reserved for evaluation')
        excluded_path=self.root/'training-excluded-pixel-hashes.txt'
        if not excluded_path.exists():raise ValueError('Run metadata split preparation before loading')
        excluded=set(excluded_path.read_text().splitlines()) if usage=='train' else set()
        seen=set()
        for manifest in sorted((self.root/'manifests').glob('*.csv')):
            if manifest.stem not in allowed:continue
            with manifest.open() as stream:
                for row in csv.DictReader(stream):
                    expected='train_candidate' if usage=='train' else 'evaluation_only'
                    if row['usage']!=expected:continue
                    h=row['pixel_sha256']
                    if h in excluded or (deduplicate and h in seen):continue
                    self.rows.append(row);seen.add(h)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];shard=row['shard']
        if shard not in self._tars:self._tars[shard]=tarfile.open(self.root/shard)
        data=self._tars[shard].extractfile(row['member']).read()
        image=Image.open(io.BytesIO(data)).convert('L')
        if self.transform:image=self.transform(image)
        return image,int(row['class_id'])
    def close(self):
        for tar in self._tars.values():tar.close()
        self._tars={}
    def __getstate__(self):
        state=self.__dict__.copy();state['_tars']={};return state
