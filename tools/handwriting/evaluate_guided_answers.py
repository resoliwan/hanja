"""Fixed-reference relation precision/recall, ablations and order stress tests.

Scores concern input→SVG edges, NOT point labels or exact split boundaries.
Unreviewed joined cases are excluded, never silently counted as correct.
"""

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

from paths import RESULTS
from stroke_mapping.guided_matcher import GuidedConfig,match_guided


def relation_metrics(cases):
    tp=fp=fn=exact=0
    for expected,predicted in cases:
        tp+=len(expected&predicted); fp+=len(predicted-expected); fn+=len(expected-predicted)
        exact+=expected==predicted
    return dict(samples=len(cases),expected_relations=tp+fn,predicted_relations=tp+fp,
                tp=tp,fp=fp,fn=fn,
                precision_pct=100*tp/(tp+fp) if tp+fp else 0.0,
                recall_pct=100*tp/(tp+fn) if tp+fn else 0.0,
                f1_pct=200*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
                exact_sample_pct=100*exact/len(cases) if cases else 0.0)


def _edges(row,field):
    return {(m['source_stroke_number'],m['svg_stroke_number']) for m in row[field]}


def _score(rows, refs):
    result={}
    for title,field in [('automatic_output','mappings'),('all_candidates','proposed_mappings')]:
        cases=[(set(map(tuple,r['expected_relations'])),_edges(rows[r['sample_id']],field)) for r in refs]
        result[title]=relation_metrics(cases)
        result[title]['by_character']={c:relation_metrics([case for r,case in zip(refs,cases) if r['character']==c])
                                      for c in dict.fromkeys(r['character'] for r in refs)}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=RESULTS/'stroke-mapping-guided-v4')
    args=parser.parse_args(); directory=args.directory
    rows={r['sample_id']:r for r in map(json.loads,(directory/'answers.jsonl').read_text().splitlines())}
    refs=[json.loads(s) for s in (directory/'relation-reference.jsonl').read_text().splitlines()]
    templates=json.loads((directory/'templates.json').read_text())
    config=GuidedConfig(**json.loads((directory/'summary.json').read_text())['config'])
    unique={}
    for ref in refs:
        if ref['trajectory_sha256']!=rows[ref['sample_id']]['trajectory_sha256']:
            raise ValueError('Stale reference trajectory')
        unique.setdefault(ref['trajectory_sha256'],ref)
    unique_refs=list(unique.values())
    baseline={r['sample_id']:r for r in map(json.loads,(RESULTS/'stroke-mapping-segments-v3/answers.jsonl').read_text().splitlines())}
    variants={'v3_local':baseline,'v4_combined':rows}
    for name,order,anchor in [('blocks_only',False,False),('blocks_order',True,False),('blocks_anchor',False,True)]:
        selected={}
        for ref in refs:
            row=rows[ref['sample_id']]
            selected[row['sample_id']]=match_guided(row['character'],row['strokes'],templates[row['character']],
                 replace(config,use_order_prior=order,use_anchor_alignment=anchor))
        variants[name]=selected
    scores={name:_score(selected,unique_refs) for name,selected in variants.items()}
    stressed_rows,stressed_refs={},[]
    for ref in unique_refs:
        row=rows[ref['sample_id']]; count=len(row['strokes'])
        result=match_guided(row['character'],[list(reversed(s)) for s in reversed(row['strokes'])],templates[row['character']],config)
        stressed_rows[row['sample_id']]=result
        stressed_refs.append(dict(ref,expected_relations=[[count+1-i,j] for i,j in ref['expected_relations']]))
    summary=dict(
        reference=dict(records=len(refs),unique_trajectories=len(unique_refs),
                       unique_relations=sum(len(r['expected_relations']) for r in unique_refs),
                       unscored_records=len(rows)-len(refs),
                       source='assistant visual review; not independent human ground truth',
                       limitations='Reviewed one-to-one input relations only; joined input boundaries are not scored. These development examples were already inspected.'),
        primary_metric='automatic_output: abstentions contribute false negatives; micro-average on unique trajectories',
        deduplicated=scores,
        record_weighted={name:_score(selected,refs) for name,selected in variants.items() if name in ('v3_local','v4_combined')},
        reversed_order_and_paths=_score(stressed_rows,stressed_refs),
        all_77_alignment_selection=dict(Counter(r.get('alignment',{}).get('mode') for r in rows.values())),
        all_77_center_fallback=sum(r.get('center_fallback',False) for r in rows.values()),
        all_77_order_overrides=sum(r.get('order_prior_overrides',0) for r in rows.values()),
    )
    (directory/'precision-recall.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    compact={name:{kind:{k:v for k,v in data.items() if k!='by_character'} for kind,data in score.items()} for name,score in scores.items()}
    print(json.dumps(dict(reference=summary['reference'],scores=compact,
                          alignments=summary['all_77_alignment_selection']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
