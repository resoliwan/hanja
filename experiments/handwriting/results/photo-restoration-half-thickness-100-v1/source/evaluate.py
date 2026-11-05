"""End-to-end evaluation against the frozen mobile ONNX classifier."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import torch

from paths import APP_ASSETS
from .synthesis import CONDITIONS, synthesize_photo


def normalize_ink(gray):
    """Pillow implementation of the app's bounding box / 97px / [-1,1] contract."""
    yy, xx = np.nonzero(gray < 210)
    result = np.full((128, 128), -1, dtype=np.float32)
    if not len(xx):
        return result
    crop = Image.fromarray(gray[yy.min():yy.max() + 1, xx.min():xx.max() + 1])
    ratio = 97 / max(crop.size)
    width, height = (max(1, int(n * ratio + .5)) for n in crop.size)
    resized = np.asarray(crop.resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32)
    x, y = (128 - width) // 2, (128 - height) // 2
    result[y:y + height, x:x + width] = 1 - resized / 127.5
    return result


def _binary(gray, auto_polarity):
    hist = np.bincount(gray.ravel(), minlength=256)
    counts = np.cumsum(hist).astype(float)
    sums = np.cumsum(hist * np.arange(256))
    other = counts[-1] - counts
    valid = (counts > 0) & (other > 0)
    scores = np.full(256, -1.0)
    scores[valid] = counts[valid] * other[valid] * (
        sums[valid] / counts[valid] - (sums[-1] - sums[valid]) / other[valid]
    ) ** 2
    dark = gray <= int(scores.argmax())
    if auto_polarity:
        border = np.concatenate((dark[0], dark[-1], dark[1:-1, 0], dark[1:-1, -1]))
        if border.mean() > .5:
            dark = ~dark
    return np.where(dark, 0, 255).astype(np.uint8)


class FrozenHanjaEvaluator:
    def __init__(self):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(APP_ASSETS / 'hanja.onnx'), options, providers=['CPUExecutionProvider'])
        self.classes = json.loads((APP_ASSETS / 'classes.json').read_text())

    def recognize(self, tensor, label):
        scores = self.session.run(None, {'image': tensor[None, None]})[0].ravel()
        order = np.argsort(-scores)
        return dict(rank=int(np.flatnonzero(order == label)[0]) + 1,
            top10=[self.classes[i] for i in order[:10]])

    def compare(self, photo, target, restored, label):
        rgb = np.uint8(np.clip(photo.transpose(1, 2, 0), 0, 1) * 255)
        gray = np.asarray(Image.fromarray(rgb).convert('L'))
        # The 0.5 restoration cutoff is fixed before looking at test results.
        ink = np.where(restored >= .5, restored, 0)
        variants = {
            'clean-target': normalize_ink(np.uint8((1 - target) * 255)),
            'gray-only': normalize_ink(gray),
            'current-binary': normalize_ink(_binary(gray, False)),
            'auto-binary': normalize_ink(_binary(gray, True)),
            'restored': normalize_ink(np.uint8(np.clip(1 - ink, 0, 1) * 255)),
        }
        return {name: self.recognize(tensor, label) for name, tensor in variants.items()}


def summarize(rows):
    summary = {}
    for condition in ['all'] + sorted({r['condition'] for r in rows}):
        group = rows if condition == 'all' else [r for r in rows if r['condition'] == condition]
        summary[condition] = {}
        for method in group[0]['methods']:
            ranks = [r['methods'][method]['rank'] for r in group]
            summary[condition][method] = dict(n=len(ranks), top1=float(np.mean(np.array(ranks) == 1)), top10=float(np.mean(np.array(ranks) <= 10)))
    return summary


@torch.inference_mode()
def evaluate_split(model, output, split, records, seed, classifier, all_conditions=False):
    clean = np.load(output / f'{split}.npy', mmap_mode='r')
    rows, samples, dice_values = [], [], []
    tasks = [(i, c) for i in range(len(records)) for c in (CONDITIONS if all_conditions else (CONDITIONS[i % len(CONDITIONS)],))]
    model.eval()
    for start in range(0, len(tasks), 16):
        batch = tasks[start:start + 16]
        pairs = [synthesize_photo(clean[i], seed + i * 31 + CONDITIONS.index(c), c) for i, c in batch]
        photos = np.stack([p for p, _ in pairs])
        prediction = model(torch.from_numpy(photos).to('mps')).sigmoid().cpu().numpy()[:, 0]
        for (i, condition), (photo, target), restored in zip(batch, pairs, prediction):
            row = records[i]
            truth = target[0]
            hard = restored >= .5
            dice = (2 * (hard * truth).sum() + 1) / (hard.sum() + truth.sum() + 1)
            dice_values.append(float(dice))
            methods = classifier.compare(photo, truth, restored, int(row['class_id']))
            rows.append(dict(sample_id=row['sample_id'], character=row['character'], class_id=int(row['class_id']), source=row['source'], condition=condition, dice=float(dice), methods=methods))
            if len(samples) < 12 and (len(rows) % 7 == 0):
                samples.append((photo, truth, restored, f"{condition} rank={methods['restored']['rank']}"))
        if start % 512 == 0:
            print(f'EVALUATE {split} {start}/{len(tasks)}', flush=True)
    _save_examples(samples, output / f'{split}-examples.png')
    return dict(summary=summarize(rows), mean_dice=float(np.mean(dice_values)), rows=rows)


def _save_examples(samples, path):
    sheet = Image.new('RGB', (3 * 160, len(samples) * 180), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (photo, target, restored, label) in enumerate(samples):
        arrays = [np.uint8(np.clip(photo.transpose(1, 2, 0), 0, 1) * 255), np.uint8((1 - target) * 255), np.uint8(np.clip(1 - restored, 0, 1) * 255)]
        for j, array in enumerate(arrays):
            sheet.paste(Image.fromarray(array).convert('RGB').resize((160, 160)), (j * 160, i * 180))
        draw.text((4, i * 180 + 162), label, fill='black')
    sheet.save(path)


@torch.inference_mode()
def evaluate_known_photos(model, output, classifier):
    """Diagnostic only: these two characters have already been inspected repeatedly."""
    directory = output.parent / 'photo-color-probe-20261104'
    manifest = json.loads((directory / 'manifest.json').read_text())
    rows, samples = [], []
    model.eval()
    for row in manifest:
        image = Image.open(row['path']).convert('RGB').resize((256, 256), Image.Resampling.BICUBIC)
        photo = np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255
        restored = model(torch.from_numpy(photo[None]).to('mps')).sigmoid().cpu().numpy()[0, 0]
        label = classifier.classes.index(row['character'])
        methods = classifier.compare(photo, np.zeros((256, 256), dtype=np.float32), restored, label)
        del methods['clean-target']
        rows.append(dict(id=row['id'], character=row['character'], condition=row['case'], methods=methods))
        if row['case'] == 'original' or 'center-glare' in row['case']:
            samples.append((photo, np.zeros((256, 256)), restored, f"{row['id']} rank={methods['restored']['rank']}"))
    _save_examples(samples, output / 'known-photo-examples.png')
    return dict(summary=summarize(rows), rows=rows, scope='Known two-glyph diagnostic, not independent real-photo evaluation')
