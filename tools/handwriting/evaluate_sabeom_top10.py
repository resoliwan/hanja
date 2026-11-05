"""Measure top-10 recall on the exact held-out set used by the Sabeom run."""
import csv
import json

import numpy as np
import torch

from hanja3500_dataset import Hanja3500Dataset
from paths import APP_ASSETS, DATASETS, RESULTS
import train_hanja3500_head as shared

DATA = DATASETS / 'korean-hanja5000-v1'
RUN = RESULTS / 'hanja5000-head-only-v1'


def main():
    torch.set_num_threads(4)
    settings = json.loads((RUN / 'settings.json').read_text())
    result = json.loads((RUN / 'result.json').read_text())
    checkpoint = RUN / 'head-only-best.pt'
    checkpoint_sha = shared.sha(checkpoint)
    assert checkpoint_sha == result['checkpoint_sha256']
    exported = json.loads((APP_ASSETS / 'export_validation.json').read_text())
    assert exported['checkpoint_sha256'] == checkpoint_sha
    for path in (DATA / 'manifests').glob('*.csv'):
        assert shared.sha(path) == settings['manifests'][path.name]
    candidates = Hanja3500Dataset(DATA, 'train')
    evaluation = Hanja3500Dataset(DATA, 'evaluation')
    rows = evaluation.rows
    assert [r['sample_id'] for r in rows] == (RUN / 'evaluation-sample-ids.txt').read_text().splitlines()
    features = np.memmap(RUN / 'features.f32', dtype='float32', mode='r', shape=(len(candidates)+len(rows), 320))
    payload = shared.load_checkpoint(checkpoint)
    classes = payload['classes']
    assert classes == evaluation.classes == json.loads((APP_ASSETS / 'classes.json').read_text())
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = shared.build_model(payload['config'], len(classes))
    model.load_state_dict(payload['ema'], strict=True)
    head = model.classifier.to(device).eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(rows), 1024):
            batch = np.array(features[len(candidates)+start:len(candidates)+start+1024], copy=True)
            chunks.append(head(torch.from_numpy(batch).to(device)).topk(10, dim=1).indices.cpu().numpy())
    top = np.concatenate(chunks)
    labels = np.array([int(r['class_id']) for r in rows])
    previous = list(csv.DictReader((RUN / 'after-predictions.csv').open(encoding='utf-8-sig')))
    assert len(previous) == len(rows)
    for row, prior, ids, label in zip(rows, previous, top, labels):
        assert row['sample_id'] == prior['sample_id']
        assert (ids[0] == label) == (prior['top1'] == 'True')
        assert (label in ids[:5]) == (prior['top5'] == 'True')
    def metrics(mask):
        ys, predictions = labels[mask], top[mask]
        values = dict(samples=len(ys), classes=len(set(ys.tolist())))
        for k in [1, 5, 10]:
            correct = (predictions[:, :k] == ys[:, None]).any(axis=1)
            values[f'top{k}'] = float(correct.mean()) if len(ys) else None
            values[f'correct_top{k}'] = int(correct.sum())
        return values
    old = set((DATASETS / 'korean-hanja3500-v1/classes.txt').read_text().splitlines())
    existing = np.array([r['character'] in old for r in rows])
    groups = {
        'scut_handwriting': np.array([r['source'] == 'scut' for r in rows]),
        'font_base': np.array([r['source'] == 'fonts' and r['variant'].endswith(':base') for r in rows]),
        'font_variants': np.array([r['source'] == 'fonts' and not r['variant'].endswith(':base') for r in rows]),
    }
    summary = {}
    for name, mask in groups.items():
        summary[name] = metrics(mask)
        summary[name]['existing_characters'] = metrics(mask & existing)
        summary[name]['new_characters'] = metrics(mask & ~existing)
    with (RUN / 'after-top10-predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'character', 'source', 'variant', 'top1', 'top5', 'top10', 'candidates'])
        for row, ids, label in zip(rows, top, labels):
            writer.writerow([row['sample_id'], row['character'], row['source'], row['variant'],
                             ids[0] == label, label in ids[:5], label in ids, ' '.join(classes[int(i)] for i in ids)])
    report = dict(checkpoint_sha256=checkpoint_sha, evaluation_samples=len(rows),
                  matches_previous_top1_top5=True, matches_app_checkpoint=True,
                  method='Frozen feature cache and saved classifier; exact prior evaluation sample order.',
                  metrics=summary)
    (RUN / 'top10-evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
