"""Extend the existing dataset without regenerating or changing its samples."""
import collections
import csv
import json
import os
import unicodedata
from pathlib import Path

import build_korean_hanja_dataset as builder
from hanja3500_dataset import Hanja3500Dataset
from paths import DATASETS, RESULTS

BASE = DATASETS / 'korean-hanja3500-v1'
OUT = DATASETS / 'korean-hanja5000-v1'


def main():
    OUT.mkdir(exist_ok=False)
    (OUT / 'manifests').mkdir()
    (OUT / 'shards').mkdir()
    old_map = json.loads((BASE / 'class-map.json').read_text())
    classes = (RESULTS / 'sabeom5000-coverage/official-characters.txt').read_text().splitlines()
    assert len(classes) == 4998 and set(old_map) < set(classes)
    metadata = {c: dict(class_id=i, grade=old_map[c]['grade'] if c in old_map else '사범') for i, c in enumerate(classes)}
    added = {c: m for c, m in metadata.items() if c not in old_map}
    (OUT / 'classes.txt').write_text('\n'.join(classes)+'\n')
    (OUT / 'class-map.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    builder.OUT = OUT
    counts = collections.defaultdict(collections.Counter)
    builder.build_fonts(added, counts)
    builder.build_aifree(added, counts, lambda c: unicodedata.normalize('NFC', c))
    builder.build_scut(added, counts, lambda c: unicodedata.normalize('NFC', c))
    (OUT / 'base-shards').mkdir()
    for source in (BASE / 'shards').glob('*.tar'):
        os.link(source, OUT / 'base-shards' / source.name)
    all_rows = []
    for source in ['ai-free', 'fonts', 'scut']:
        path = OUT / 'manifests' / f'{source}.csv'
        with path.open() as stream:
            additions = list(csv.DictReader(stream))
        for row in additions:
            row['sample_id'] = 'sabeom-' + row['sample_id']
        with (BASE / 'manifests' / f'{source}.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row['class_id'] = metadata[row['character']]['class_id']
            row['shard'] = 'base-shards/' + Path(row['shard']).name
        rows += additions
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=builder.FIELDS)
            writer.writeheader(); writer.writerows(rows)
        all_rows.extend(rows)
    # A pixel shared across train/evaluation, or across labels, cannot train.
    hashes = collections.defaultdict(set)
    evaluation_hashes = set()
    for row in all_rows:
        hashes[row['pixel_sha256']].add(row['character'])
        if row['usage'] == 'evaluation_only':
            evaluation_hashes.add(row['pixel_sha256'])
    conflicts = {h for h, labels in hashes.items() if len(labels) > 1}
    excluded = conflicts | evaluation_hashes
    (OUT / 'training-excluded-pixel-hashes.txt').write_text('\n'.join(sorted(excluded))+'\n')
    candidates = Hanja3500Dataset(OUT, 'train')
    evaluation = Hanja3500Dataset(OUT, 'evaluation')
    old_valid = set((RESULTS / 'hanja3500-head-only-v1/validation-sample-ids.txt').read_text().splitlines())
    validation = {r['sample_id'] for r in candidates.rows if r['sample_id'] in old_valid}
    by_char = collections.defaultdict(list)
    for row in candidates.rows:
        if row['character'] in added and row['source'] == 'ai-free':
            by_char[row['character']].append(row)
    for rows in by_char.values():
        rows.sort(key=lambda r: r['pixel_sha256'])
        # Deterministic, pixel-disjoint split; writer independence is unknown.
        n = min(len(rows)-1, max(1, round(len(rows)*0.08)))
        validation.update(r['sample_id'] for r in rows[:n])
    (OUT / 'validation-sample-ids.txt').write_text('\n'.join(sorted(validation))+'\n')
    train = [r for r in candidates.rows if r['sample_id'] not in validation]
    valid = [r for r in candidates.rows if r['sample_id'] in validation]
    assert len({r['character'] for r in train}) == len(classes)
    assert not {r['pixel_sha256'] for r in train} & {r['pixel_sha256'] for r in valid+evaluation.rows}
    assert {r['character'] for r in evaluation.rows if r['source']=='fonts'} == set(classes)
    fonts = json.loads((BASE / 'font-coverage.json').read_text())
    new_fonts = json.loads((OUT / 'font-coverage.json').read_text())
    for old, new in zip(fonts, new_fonts):
        assert old['font'] == new['font']
        for key in ['covered_characters', 'missing_characters']:
            new[key] = sorted(set(old[key]) | set(new[key]))
        new['covered'] = len(new['covered_characters']); new['missing'] = len(new['missing_characters'])
    (OUT / 'font-coverage.json').write_text(json.dumps(new_fonts, ensure_ascii=False, indent=2))
    per_char = collections.defaultdict(collections.Counter)
    for row in all_rows:
        per_char[row['character']][row['source']] += 1
    with (OUT / 'character-coverage.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['character','ai_free_images','scut_images','font_images'])
        for c in classes:
            writer.writerow([c, per_char[c]['ai-free'], per_char[c]['scut'], per_char[c]['fonts']])
    summary = dict(classes=len(classes), added_classes=len(added), train=len(train), validation=len(valid),
                   evaluation=len(evaluation), raw_images=len(all_rows), conflicting_hashes=len(conflicts),
                   source_images=dict(collections.Counter(r['source'] for r in all_rows)),
                   source_classes={s:len({r['character'] for r in all_rows if r['source']==s}) for s in ['ai-free','scut','fonts']},
                   held_out_font='noto-sans-kr', scut_usage='evaluation_only', validation_writer_independent=False,
                   training_evaluation_pixel_overlap=0, base_samples_unchanged=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
