"""Coordinate-only batch diagnostics; assignment coverage is not accuracy."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import median

from paths import RESULTS


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _diagnose(row, target_count):
    pairs = {(m['source_stroke_number'], m['svg_stroke_number']) for m in row['mappings']}
    if any(i<1 or i>len(row['strokes']) or j<1 or j>target_count for i,j in pairs):
        raise ValueError('Out-of-range correspondence: '+row['sample_id'])
    missing = sorted(set(range(1,target_count+1))-{j for _,j in pairs})
    inputs, targets = defaultdict(set), defaultdict(set)
    for i,j in pairs:
        inputs[i].add(j); targets[j].add(i)
    for i in inputs:
        spans = sorted((s['source_fraction_start'],s['source_fraction_end'])
                       for s in row['segment_mappings']
                       if s['source_stroke_number']==i and s['svg_stroke_number'] is not None)
        if any(a[1]>b[0]+1e-9 for a,b in zip(spans,spans[1:])):
            raise ValueError('Overlapping assigned intervals: '+row['sample_id'])
    status = ('invalid' if row['status']=='invalid' else
              'missing_targets' if missing else 'all_targets_assigned')
    return dict(sample_id=row['sample_id'], character=row['character'], status=status,
                processing_reasons=row.get('reasons', []),
                missing_target_strokes=missing,
                fully_unassigned_inputs=sorted(set(range(1,len(row['strokes'])+1))-set(inputs)),
                multiple_inputs_for_target=any(len(s)>1 for s in targets.values()),
                multiple_targets_for_input=any(len(s)>1 for s in inputs.values()),
                unused_ink_fraction=row['unmatched_fraction'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=RESULTS/'stroke-mapping-guided-v4/relation-reference.jsonl')
    args = parser.parse_args()
    rows = _read(args.directory/'answers.jsonl')
    summary = json.loads((args.directory/'summary.json').read_text())
    templates = json.loads((args.directory/'templates.json').read_text())
    diagnostics = [_diagnose(r,len(templates[r['character']])) for r in rows]
    counts = Counter(d['status'] for d in diagnostics)
    by_character = []
    for char in templates:
        selected = [d for d in diagnostics if d['character']==char]
        cc = Counter(d['status'] for d in selected)
        by_character.append(dict(character=char, svg_strokes=len(templates[char]), total=len(selected),
                                 unique_trajectories=len({r['trajectory_sha256'] for r in rows if r['character']==char}),
                                 all_targets_assigned=cc['all_targets_assigned'],
                                 missing_targets=cc['missing_targets'], invalid=cc['invalid'],
                                 multiple_inputs_for_target=sum(d['multiple_inputs_for_target'] for d in selected),
                                 multiple_targets_for_input=sum(d['multiple_targets_for_input'] for d in selected)))
    # Reuse only existing annotations. No new visual inspection or index-based
    # pseudo-ground-truth is introduced for the additional grade characters.
    references = _read(args.reference) if args.reference.exists() else []
    by_id = {r['sample_id']:r for r in rows}
    evaluated, seen = [], set()
    for ref in references:
        row = by_id.get(ref['sample_id'])
        if row is None:
            continue
        if ref['trajectory_sha256']!=row['trajectory_sha256']:
            raise ValueError('Stale evaluation reference')
        if ref['trajectory_sha256'] in seen:
            continue
        seen.add(ref['trajectory_sha256'])
        expected = set(map(tuple,ref['expected_relations']))
        predicted = {(m['source_stroke_number'],m['svg_stroke_number']) for m in row['mappings']}
        evaluated.append(dict(sample_id=row['sample_id'],character=row['character'],
                              correct=expected==predicted,missing=sorted(expected-predicted),extra=sorted(predicted-expected)))
    times = sorted(r['runtime_seconds'] for r in rows)
    unique = len({r['trajectory_sha256'] for r in rows})
    result = dict(grade=summary['selection']['grade'], grade_system=summary['selection']['grade_system'],
                  total_characters=len(templates), total_records=len(rows), unique_trajectories=unique,
                  duplicate_records=len(rows)-unique, status_counts=dict(counts), by_character=by_character,
                  multiple_inputs_for_target=sum(d['multiple_inputs_for_target'] for d in diagnostics),
                  multiple_targets_for_input=sum(d['multiple_targets_for_input'] for d in diagnostics),
                  records_with_fully_unassigned_inputs=sum(bool(d['fully_unassigned_inputs']) for d in diagnostics),
                  runtime_seconds=dict(total=sum(times),median=median(times),p95=times[min(len(times)-1,int(.95*len(times)))],maximum=max(times)),
                  existing_reference_evaluation=dict(total=len(evaluated),correct=sum(r['correct'] for r in evaluated),
                                                     incorrect=sum(not r['correct'] for r in evaluated),samples=evaluated),
                  accuracy_for_full_batch=None,
                  note='No new visual inspection. Complete assignment is not verified correctness. Existing reference subset is assistant-reviewed development data.')
    (args.directory/'processing-report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (args.directory/'diagnostics.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in diagnostics))
    with (args.directory/'by-character.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(by_character[0]))
        writer.writeheader(); writer.writerows(by_character)
    lines = [f"# {result['grade']}급 SVG 획 매핑 실행 결과", '',
             f"Simpledic 대한검정회 기준 {len(templates)}자, SCUT 필기 {len(rows)}건. 고유 궤적 {unique}건, 중복 {len(rows)-unique}건.",
             '기존 v5 설정과 원본 SVG를 그대로 사용했다. 카메라·이미지 판독·새 시각적 검수는 수행하지 않았고 HTML/PNG도 생성하지 않았다.', '',
             '| 자동 처리 상태 | 필기 수 |', '|---|---:|',
             f"| 모든 SVG 획에 입력 구간 배정 | {counts['all_targets_assigned']} |",
             f"| SVG 획 누락 있음 | {counts['missing_targets']} |",
             f"| 현재 알고리즘이 처리하지 못한 입력 | {counts['invalid']} |", '',
             '**모든 획에 배정했다는 것은 정답이라는 뜻이 아니다.** 새 정답 주석이 없으므로 전체 정답/오류 건수와 정밀도·재현율은 산출하지 않는다. 누락은 좌표 결과로 확인한 구조적 문제다.', '',
             '| 글자 | SVG 획수 | 필기 수 | 전체 획 배정 | 획 누락 | 처리 불가 |', '|---|---:|---:|---:|---:|---:|']
    for r in by_character:
        lines.append(f"| {r['character']} | {r['svg_strokes']} | {r['total']} | {r['all_targets_assigned']} | {r['missing_targets']} | {r['invalid']} |")
    lines += ['', f"여러 입력을 한 SVG 획에 묶은 사례 {result['multiple_inputs_for_target']}건, 한 입력을 여러 SVG 획에 나눈 사례 {result['multiple_targets_for_input']}건. 두 집계는 중복될 수 있다.",
              f"아무 SVG 획에도 배정되지 않은 펜 입력이 있는 사례 {result['records_with_fully_unassigned_inputs']}건. 이 값만으로 해당 입력이 오류인지 연결 동작인지 판정하지 않는다.", '',
              '한 점 또는 길이 0인 획은 현재 경로 정규화가 지원하지 않는다. 처리 불가를 사용자가 글자를 잘못 썼다는 뜻으로 해석하지 않는다.', '',
              '## 기존 정답 주석과 겹치는 부분', '',
              f"기존 주석을 변경 없이 재사용한 고유 필기 {len(evaluated)}건: 정답 {sum(r['correct'] for r in evaluated)}건, 오류 {sum(not r['correct'] for r in evaluated)}건. 새 시각적 평가를 한 것이 아니며 전체 8급의 정확도로 일반화하지 않는다.", '',
              '## 실행', '',
              '```sh', '.venv/bin/python tools/handwriting/build_stroke_answers.py --method target --grade 8 --no-visuals',
              '.venv/bin/python tools/handwriting/summarize_target_answers.py --directory experiments/handwriting/results/stroke-mapping-target-grade8-v5', '```', '',
              '원본 좌표와 구간별 배정은 answers.jsonl, 필기별 진단은 diagnostics.jsonl, 글자별 집계는 by-character.csv에 기록했다. 실행 설정·목록 출처·SVG 및 원본 데이터 해시는 summary.json에 있다.', '']
    (args.directory/'processing-report.md').write_text('\n'.join(lines))
    print(json.dumps({k:result[k] for k in ('total_characters','total_records','unique_trajectories','status_counts','multiple_inputs_for_target','multiple_targets_for_input','runtime_seconds')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
