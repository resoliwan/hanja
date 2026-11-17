"""Audit v3, separating coverage acceptance from correctness evidence."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from paths import RESULTS
from stroke_mapping.segment_matcher import SegmentConfig, match_segments


def _target_sets(row):
    return [sorted({m['svg_stroke_number'] for m in row['proposed_mappings'] if m['source_stroke_number']==i+1})
            for i in range(len(row['strokes']))]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=RESULTS/'stroke-mapping-segments-v3')
    args=parser.parse_args(); directory=args.directory
    rows=[json.loads(s) for s in (directory/'answers.jsonl').read_text().splitlines()]
    templates=json.loads((directory/'templates.json').read_text())
    config=SegmentConfig(**json.loads((directory/'summary.json').read_text())['config'])
    baseline_dir=RESULTS/'stroke-mapping-123-yong-v2'
    previous={r['sample_id']:r for r in map(json.loads,(baseline_dir/'answers.jsonl').read_text().splitlines())}
    reviews={r['sample_id']:r for r in map(json.loads,(baseline_dir/'visual-assessments.jsonl').read_text().splitlines())}
    comparisons,probes=[],[]
    for row in rows:
        sid=row['sample_id']; baseline=previous.get(sid)
        if baseline:
            review=reviews[sid]
            if review['trajectory_sha256']!=row['trajectory_sha256']:
                raise ValueError('Review does not match current trajectory')
            expected=review['expected_svg_numbers']
            actual=_target_sets(row)
            comparisons.append(dict(sample_id=sid,character=row['character'],
                previous_status=baseline['status'],status=row['status'],
                reviewed_one_to_one=expected is not None,
                expected_target_sets=[[n] for n in expected] if expected else None,
                proposed_target_sets=actual,
                target_sets_agree=actual==[[n] for n in expected] if expected else None))
        # Match the same physical input after reversing both input and point
        # order. Compare target SETS, not the arbitrary numbering of runs.
        source=[list(reversed(s)) for s in reversed(row['strokes'])]
        reversed_result=match_segments(row['character'],source,templates[row['character']],config)
        reversed_result['strokes']=source
        reverse_agrees=(_target_sets(reversed_result)==list(reversed(_target_sets(row)))
                        and reversed_result['status']==row['status'])
        finer=match_segments(row['character'],row['strokes'],templates[row['character']],
                             replace(config,segment_length=config.segment_length/2))
        finer['strokes']=row['strokes']
        probes.append(dict(sample_id=sid,reverse_agrees=reverse_agrees,
                          half_step_target_sets_agree=_target_sets(finer)==_target_sets(row),
                          half_step_status_agrees=finer['status']==row['status']))
    reviewed=[r for r in comparisons if r['reviewed_one_to_one']]
    result=dict(
        total=len(rows),common_baseline_samples=len(comparisons),
        baseline_accepted=sum(r['previous_status']=='accepted' for r in comparisons),
        current_accepted_on_common_samples=sum(r['status']=='accepted' for r in comparisons),
        newly_accepted=[r['sample_id'] for r in comparisons if r['previous_status']=='pending' and r['status']=='accepted'],
        newly_pending=[r['sample_id'] for r in comparisons if r['previous_status']=='accepted' and r['status']=='pending'],
        previously_visually_reviewed_one_to_one=len(reviewed),
        target_sets_agree_with_previous_review=sum(r['target_sets_agree'] for r in reviewed),
        target_set_disagreements=[r for r in reviewed if not r['target_sets_agree']],
        reverse_checks=len(probes),reverse_agree=sum(r['reverse_agrees'] for r in probes),
        half_step_target_sets_agree=sum(r['half_step_target_sets_agree'] for r in probes),
        half_step_status_agree=sum(r['half_step_status_agrees'] for r in probes),
        accuracy=None,
        limitation='Target-set agreement does not prove correct interval boundaries or connector labels. Prior annotations are assistant visual review, not independent human gold.',
    )
    (directory/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (directory/'resolution-and-order-checks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in probes))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
