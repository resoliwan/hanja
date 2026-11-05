"""Measure predicted ink on unseen procedural blank backgrounds after training."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from paths import RESULTS
from .model import PhotoInkRestorer
from .synthesis import CONDITIONS, synthesize_photo


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=RESULTS / 'photo-restoration-pilot-v1')
    output = parser.parse_args().output
    torch.set_num_threads(4)
    model = PhotoInkRestorer().to('mps').eval()
    model.load_state_dict(torch.load(output / 'best.pt', map_location='cpu', weights_only=False)['model'])
    blank = np.full((128, 128), 255, np.uint8)
    rows = []
    for condition in CONDITIONS:
        for start in range(0, 100, 10):
            images = [synthesize_photo(blank, 1_100_000_001 + 1000 * CONDITIONS.index(condition) + i, condition, blank=True)[0] for i in range(start, start + 10)]
            probability = model(torch.from_numpy(np.stack(images)).to('mps')).sigmoid().cpu().numpy()
            for p in probability:
                rows.append(dict(condition=condition, foreground_pixels=int((p >= .5).sum()), max_probability=float(p.max())))
    result = dict(n=len(rows), any_foreground_rate=float(np.mean([r['foreground_pixels'] > 0 for r in rows])),
        at_least_16_pixels_rate=float(np.mean([r['foreground_pixels'] >= 16 for r in rows])),
        scope='Unseen procedural backgrounds only. Diagnostic area threshold, not calibrated rejection policy.', rows=rows)
    (output / 'blank-results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
