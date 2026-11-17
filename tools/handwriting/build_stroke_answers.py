"""Build auditable offline SCUT→SVG stroke answers; segment mode adds 四."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from paths import DATASETS, RESULTS, ROOT
from stroke_mapping.matcher import MappingConfig, SUPPORTED_CHARACTERS, match_strokes
from stroke_mapping.report import summarize, write_overview, write_report, write_visual_sheets
from stroke_mapping.sources import read_grade_entries, read_scut, read_svg_strokes
from stroke_mapping.segment_matcher import SegmentConfig, match_segments, SUPPORTED_CHARACTERS as SEGMENT_CHARACTERS
from stroke_mapping.segment_report import write_segment_report
from stroke_mapping.guided_matcher import GuidedConfig, match_guided
from stroke_mapping.target_matcher import TargetMappingConfig, match_targets


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scut', type=Path, default=DATASETS / 'onHCCTestDB-TradChar.zip')
    parser.add_argument('--dictionary-dir', type=Path, default=ROOT.parent / 'simpledic/data/dictionaries/hanja-ko')
    parser.add_argument('--svg-dir', type=Path)
    parser.add_argument('--method', choices=('target', 'guided', 'segments', 'whole'), default='target')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--characters')
    selection.add_argument('--grade', help='Simpledic 대한검정회 신출 급수, 예: 8')
    parser.add_argument('--no-visuals', action='store_true', help='Write coordinate results only; no HTML or PNG rendering')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.svg_dir is None:
        args.svg_dir = args.dictionary_dir / 'assets/stroke-order'
    grade_entries = None
    if args.grade is not None:
        if args.method!='target':
            parser.error('--grade requires --method target')
        grade_entries = read_grade_entries(args.dictionary_dir, args.grade)
        args.characters = ''.join(e['character'] for e in grade_entries)
    supported = SUPPORTED_CHARACTERS if args.method == 'whole' else SEGMENT_CHARACTERS
    if args.characters is None:
        args.characters = supported
    if not args.characters or (args.method!='target' and set(args.characters) - set(supported)):
        parser.error('Supported characters: '+supported)
    if args.out is None:
        args.out = RESULTS / {'target':'stroke-mapping-target-v5','guided':'stroke-mapping-guided-v4','segments':'stroke-mapping-segments-v3','whole':'stroke-mapping-123-yong-v2'}[args.method]
        if args.grade is not None:
            args.out = RESULTS / f'stroke-mapping-target-grade{args.grade}-v5'
    config = {'target':TargetMappingConfig,'guided':GuidedConfig,'segments':SegmentConfig,'whole':MappingConfig}[args.method]()
    matcher = {'target':match_targets,'guided':match_guided,'segments':match_segments,'whole':match_strokes}[args.method]
    templates, provenance = {}, {}
    for character in dict.fromkeys(args.characters):
        path = args.svg_dir / f'u{ord(character):04x}.svg'
        templates[character] = read_svg_strokes(path)
        provenance[character] = dict(path=str(path.resolve()), sha256=_sha256(path))
    if grade_entries:
        for entry in grade_entries:
            if len(templates[entry['character']]) != entry['declared_strokes']:
                raise ValueError('SVG stroke count differs from grade entry: '+entry['character'])
    rows = []
    for sample in read_scut(args.scut, args.characters):
        sample['trajectory_sha256'] = hashlib.sha256(json.dumps(
            [sample['character'], sample['strokes']], separators=(',', ':')).encode()).hexdigest()
        started = time.perf_counter()
        result = matcher(sample['character'], sample['strokes'], templates[sample['character']], config)
        result['runtime_seconds'] = time.perf_counter()-started
        rows.append(dict(**sample, **result))
        if len(rows)%25==0:
            print(f'Mapped {len(rows)} records', flush=True)
    if {r['character'] for r in rows} != set(args.characters):
        raise ValueError('SCUT archive must contain all requested characters')
    summary = summarize(rows)
    summary.update(
        algorithm={'target':'target-bundle-exclusive-v5','guided':'direction-block-order-anchor-v4','segments':'local-segment-projection-v3','whole':'structural-correspondence-stability-v2'}[args.method],
        writing_order_used_for_assignment=args.method in ('target', 'guided'),
        stability_note=('Coverage and alignment fit are not correctness.' if args.method!='whole'
                        else 'Agreement across fixed geometric probes, not a probability of correctness.'),
        mapping_schema='one_to_one_strokes' if args.method=='whole' else 'many_to_many_source_intervals',
        config=asdict(config), source=dict(path=str(args.scut.resolve()), sha256=_sha256(args.scut)),
        templates=provenance, mapping_indices='1-based; source=actual writing order, SVG=numbered clip-path order',
        selection=dict(characters=list(dict.fromkeys(args.characters)), grade=args.grade,
                       grade_system='daehan-geomjeonghoe-hanja' if args.grade is not None else None,
                       entries=grade_entries, template_variant='original stroke-order SVGs'),
        visual_outputs=not args.no_visuals,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'answers.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows), encoding='utf-8')
    for filename, data in [('templates.json', templates), ('summary.json', summary)]:
        (args.out / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    if not args.no_visuals:
        if args.method != 'whole':
            write_segment_report(args.out, rows, summary)
        else:
            write_report(args.out / 'review.html', rows, summary)
            write_overview(args.out / 'overview.png', rows)
        write_visual_sheets(args.out, rows, templates)
    print(json.dumps(dict(total=summary['total'], status_counts=summary['status_counts'],
                          results=str(args.out.resolve())),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
