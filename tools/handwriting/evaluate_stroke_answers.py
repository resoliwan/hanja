"""Compare generated answers with visual annotations and test order invariance.

Visual annotations are fallible assistant judgments, not independent gold labels.
"""

import argparse
import json
from pathlib import Path
import random

from paths import RESULTS
from stroke_mapping.matcher import MappingConfig, match_strokes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=RESULTS / 'stroke-mapping-123-yong-v2')
    args = parser.parse_args()
    directory = args.directory
    answers = [json.loads(s) for s in (directory/'answers.jsonl').read_text().splitlines()]
    annotations = {r['sample_id']: r for r in map(json.loads, (directory/'visual-assessments.jsonl').read_text().splitlines())}
    if set(annotations) != {r['sample_id'] for r in answers}:
        raise ValueError('Annotations must cover exactly the evaluated answers')
    templates = json.loads((directory/'templates.json').read_text())
    config = MappingConfig(**json.loads((directory/'summary.json').read_text())['config'])
    evaluations, probes = [], []
    for row in answers:
        review = annotations[row['sample_id']]
        if review['trajectory_sha256'] != row['trajectory_sha256']:
            raise ValueError('Visual annotation belongs to a different trajectory')
        expected = review['expected_svg_numbers']
        predicted = [m['svg_stroke_number'] for m in row['mappings']]
        agrees = row['status'] == 'pending' if expected is None else predicted == expected
        evaluations.append(dict(sample_id=row['sample_id'], character=row['character'],
                                trajectory_sha256=row['trajectory_sha256'],
                                visually_mappable=expected is not None,
                                agrees_with_visual_review=agrees))
        if expected is None:
            continue
        for variant in ('shuffled_order', 'shuffled_and_reversed_paths'):
            order = list(range(len(row['strokes'])))
            random.Random(20261106).shuffle(order)
            if len(order) > 1 and order == list(range(len(order))):
                order.reverse()
            source = [row['strokes'][i] for i in order]
            if variant == 'shuffled_and_reversed_paths':
                source = [list(reversed(stroke)) for stroke in source]
            result = match_strokes(row['character'], source, templates[row['character']], config)
            actual = [m['svg_stroke_number'] for m in result['mappings']]
            target = [expected[i] for i in order]
            probes.append(dict(sample_id=row['sample_id'], variant=variant,
                               source_order=[i+1 for i in order], expected=target, actual=actual,
                               passed=result['status']=='accepted' and actual==target))
    unique = {r['trajectory_sha256']: r for r in evaluations}
    summary = dict(
        reviewer='assistant_visual_review_not_independent_gold',
        total=len(evaluations), unique_trajectories=len(unique),
        visually_mappable=sum(r['visually_mappable'] for r in evaluations),
        agrees_with_visual_review=sum(r['agrees_with_visual_review'] for r in evaluations),
        unique_agrees_with_visual_review=sum(r['agrees_with_visual_review'] for r in unique.values()),
        correspondence_disagreements=[r['sample_id'] for r in evaluations if not r['agrees_with_visual_review']],
        order_invariance_checks=len(probes), order_invariance_passed=sum(r['passed'] for r in probes),
        order_invariance_failures=[r for r in probes if not r['passed']],
        by_character={c: dict(total=sum(r['character']==c for r in evaluations),
                              mappable=sum(r['character']==c and r['visually_mappable'] for r in evaluations),
                              agrees=sum(r['character']==c and r['agrees_with_visual_review'] for r in evaluations))
                      for c in templates},
        limitation='一 is a unique correspondence conditional on a trusted character label. No recognition accuracy or population accuracy is claimed.',
    )
    (directory/'visual-evaluation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    (directory/'order-invariance.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in probes))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
