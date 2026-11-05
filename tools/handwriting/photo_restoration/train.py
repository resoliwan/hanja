"""Run the 500-class, 20k-source, three-epoch restoration pilot on MPS."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from paths import RESULTS, APP_ASSETS
from .evaluate import FrozenHanjaEvaluator, evaluate_known_photos, evaluate_split
from .model import PhotoInkRestorer, restoration_loss
from .prepare import prepare_pilot
from .synthesis import synthesize_photo


class _SyntheticPairs(Dataset):
    def __init__(self, path, seed):
        self.images = np.load(path, mmap_mode='r')
        self.seed = seed

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        seed = self.seed + index * 101
        photo, mask = synthesize_photo(self.images[index], seed, blank=(index % 20 == 0))
        return torch.from_numpy(photo), torch.from_numpy(mask)


def _save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=RESULTS / 'photo-restoration-pilot-v1')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    assert torch.backends.mps.is_available(), 'This pilot is configured for Apple MPS'
    if (output / 'last.pt').exists():
        raise RuntimeError('Existing training checkpoint; choose a fresh output to avoid overwriting it')
    torch.set_num_threads(4)
    torch.manual_seed(20261104)
    started = time.time()
    split = prepare_pilot(output)
    model = PhotoInkRestorer().to('mps')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    classifier = FrozenHanjaEvaluator()
    settings = dict(seed=20261104, selected_classes=500, train_sources=len(split['records']['train']), epochs=3, batch_size=16, input_size=256, parameters=sum(p.numel() for p in model.parameters()), learning_rate=.001, blank_fraction=.05, device='MPS', output_cutoff=.5, augmentation='procedural backgrounds only, mild affine + RGB + texture + glare + blur/noise; no real-photo backgrounds/JPEG in pilot', model_sha256=hashlib.sha256((APP_ASSETS / 'hanja.onnx').read_bytes()).hexdigest(), torch_version=str(torch.__version__), source_manifest_hashes=split['source_manifest_hashes'])
    _save_json(output / 'settings.json', settings)
    print('TRAIN SETTINGS', json.dumps(settings), flush=True)
    best_score = (-1.0, -1.0)
    history = []
    for epoch in range(1, 4):
        dataset = _SyntheticPairs(output / 'train.npy', 20261104 + epoch * 10_000_000)
        generator = torch.Generator().manual_seed(20261104 + epoch)
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2,
            multiprocessing_context='spawn', persistent_workers=True, generator=generator)
        model.train()
        epoch_started = time.time()
        total_loss = 0.0
        for step, (photo, target) in enumerate(loader, 1):
            photo, target = photo.to('mps'), target.to('mps')
            optimizer.zero_grad(set_to_none=True)
            loss = restoration_loss(model(photo), target)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f'Nonfinite loss at epoch {epoch}, step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            if step % 100 == 0 or step == len(loader):
                elapsed = time.time() - epoch_started
                progress = dict(epoch=epoch, step=step, steps=len(loader), loss=total_loss / step, elapsed_seconds=elapsed, estimated_epoch_remaining_seconds=elapsed / step * (len(loader) - step))
                _save_json(output / 'progress.json', progress)
                print('TRAIN', json.dumps(progress), flush=True)
        del loader
        validation = evaluate_split(model, output, 'validation', split['records']['validation'], 700_000_001, classifier)
        summary = validation['summary']['all']
        score = (summary['restored']['top1'], validation['mean_dice'])
        record = dict(epoch=epoch, train_loss=total_loss / step, seconds=time.time() - epoch_started, validation=summary, validation_dice=validation['mean_dice'])
        history.append(record)
        _save_json(output / 'history.json', history)
        _save_json(output / f'validation-epoch{epoch}.json', validation)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        checkpoint = dict(model=state, epoch=epoch, settings=settings, validation=record)
        torch.save(checkpoint, output / 'last.pt')
        if score > best_score:
            best_score = score
            torch.save(checkpoint, output / 'best.pt')
        print('VALIDATION', json.dumps(record), flush=True)
    best = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(best['model'])
    test = evaluate_split(model, output, 'test', split['records']['test'], 900_000_001, classifier, all_conditions=True)
    _save_json(output / 'test-results.json', test)
    photos = evaluate_known_photos(model, output, classifier)
    _save_json(output / 'known-photo-results.json', photos)
    result = dict(best_epoch=best['epoch'], elapsed_seconds=time.time() - started, test_summary=test['summary'], test_dice=test['mean_dice'], known_photo_summary=photos['summary']['all'])
    _save_json(output / 'result.json', result)
    print('COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
