"""Binary sample evaluation against frozen references, plus constructed splits.

Every reviewed prediction is correct or incorrect; there is no pending class.
References without independent interval labels only score input→SVG relations.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path

from paths import RESULTS
from evaluate_guided_answers import relation_metrics
from stroke_mapping.geometry import path_length
from stroke_mapping.target_matcher import TargetMappingConfig, match_targets


def _read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _edges(row, field='mappings'):
    return {(m['source_stroke_number'], m['svg_stroke_number']) for m in row[field]}


def _verdict(expected, actual):
    return dict(verdict='correct' if expected==actual else 'incorrect',
                missing=sorted(expected-actual), extra=sorted(actual-expected))


def _split_middle(points):
    halfway = path_length(points)/2
    walked = 0.
    for i, (a, b) in enumerate(zip(points, points[1:])):
        length = path_length([a,b])
        if walked+length >= halfway and length>0:
            fraction = (halfway-walked)/length
            point = tuple(a[k]+fraction*(b[k]-a[k]) for k in (0,1))
            left, right = points[:i+1]+[point], [point]+points[i+1:]
            return left, right
        walked += length
    raise ValueError('Cannot split zero-length input')


def _constructed_cases(refs, rows, templates, config):
    """One fixed seed per character, chosen without inspecting v5 results."""
    seeds = {}
    for ref in refs:
        seeds.setdefault(ref['character'], ref)
    results = []
    for char, ref in seeds.items():
        source = rows[ref['sample_id']]['strokes']
        expected = set(map(tuple, ref['expected_relations']))
        # Split the longest original input; both halves retain its SVG label.
        index = max(range(len(source)), key=lambda i: path_length(source[i]))
        left, right = _split_middle(source[index])
        split = source[:index]+[left,right]+source[index+1:]
        split_expected = set()
        for i,j in expected:
            if i==index+1:
                split_expected.update(((i,j),(i+1,j)))
            else:
                split_expected.add((i+int(i>index+1),j))
        variants = [('split_longest', split, split_expected)]
        if len(source)>1:
            # One pen-down trajectory now contains two original strokes plus
            # a synthetic connector. The connector has no SVG label.
            joined = [source[0]+source[1]]+source[2:]
            joined_expected = {(1 if i<=2 else i-1,j) for i,j in expected}
            variants.append(('join_first_two', joined, joined_expected))
        for name, input_strokes, truth in variants:
            actual = match_targets(char, input_strokes, templates[char], config)
            results.append(dict(sample_id=ref['sample_id'], character=char, transformation=name,
                                expected_relations=sorted(truth), predicted_relations=sorted(_edges(actual)),
                                **_verdict(truth, _edges(actual))))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=RESULTS/'stroke-mapping-target-v5')
    parser.add_argument('--reference-directory', type=Path, default=RESULTS/'stroke-mapping-guided-v4')
    args = parser.parse_args()
    rows = {r['sample_id']:r for r in _read_rows(args.directory/'answers.jsonl')}
    baseline = {r['sample_id']:r for r in _read_rows(args.reference_directory/'answers.jsonl')}
    references = _read_rows(args.reference_directory/'relation-reference.jsonl')
    unique = {}
    for ref in references:
        for source in (rows, baseline):
            if source[ref['sample_id']]['trajectory_sha256'] != ref['trajectory_sha256']:
                raise ValueError('Reference trajectory does not match evaluated input')
        key = ref['trajectory_sha256']
        if key in unique and unique[key]['expected_relations'] != ref['expected_relations']:
            raise ValueError('Conflicting duplicate reference labels')
        unique.setdefault(key, ref)
    cases, baseline_cases, verdicts = [], [], []
    by_character = defaultdict(Counter)
    for ref in unique.values():
        expected = set(map(tuple, ref['expected_relations']))
        predicted = _edges(rows[ref['sample_id']])
        previous = _edges(baseline[ref['sample_id']], 'proposed_mappings')
        verdict = _verdict(expected, predicted)
        verdicts.append(dict(sample_id=ref['sample_id'], character=ref['character'],
                             baseline_verdict=_verdict(expected, previous)['verdict'], **verdict))
        by_character[ref['character']][verdict['verdict']] += 1
        cases.append((expected,predicted)); baseline_cases.append((expected,previous))
    templates = json.loads((args.directory/'templates.json').read_text())
    config = TargetMappingConfig(**json.loads((args.directory/'summary.json').read_text())['config'])
    constructed = _constructed_cases(list(unique.values()), rows, templates, config)
    counts = Counter(v['verdict'] for v in verdicts)
    old_counts = Counter(v['baseline_verdict'] for v in verdicts)
    missing_targets = [dict(sample_id=r['sample_id'], character=r['character'],
                            missing=r.get('missing_target_strokes', []))
                       for r in rows.values() if r.get('missing_target_strokes')]
    result = dict(
        scope=dict(executed=len(rows), reviewed_records=len(references), unique_reviewed=len(unique),
                   unreviewed_records=len(rows)-len(references),
                   reference=str((args.reference_directory/'relation-reference.jsonl').resolve()),
                   note='Assistant-reviewed development data, not independent human gold. Exact sample correctness compares input-to-SVG relations, not interval boundaries.'),
        config=asdict(config),
        current=dict(total=len(verdicts), correct=counts['correct'], incorrect=counts['incorrect'],
                     by_character={c:dict(total=sum(n.values()),correct=n['correct'],incorrect=n['incorrect']) for c,n in by_character.items()}),
        baseline=dict(total=len(verdicts), correct=old_counts['correct'], incorrect=old_counts['incorrect']),
        improved=sum(v['baseline_verdict']=='incorrect' and v['verdict']=='correct' for v in verdicts),
        regressed=sum(v['baseline_verdict']=='correct' and v['verdict']=='incorrect' for v in verdicts),
        relation_metrics=relation_metrics(cases), baseline_relation_metrics=relation_metrics(baseline_cases),
        samples=verdicts, constructed_cases=constructed,
        constructed_summary=dict(Counter(c['verdict'] for c in constructed)),
        all_record_missing_targets=missing_targets,
    )
    (args.directory/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines = [
        '# SVG 기준 묶음 매핑 v5 평가', '',
        f"전체 실행 {len(rows)}건. 정답 주석이 있는 {len(references)}건에서 중복을 제거한 {len(unique)}건을 평가했다.",
        '필기 한 개의 입력→SVG 연결이 모두 맞아야 정답이다. 누락이나 추가 연결이 하나라도 있으면 오류다. 보류는 없다.', '',
        '| 방법 | 전체 | 정답 | 오류 |', '|---|---:|---:|---:|',
        f"| 기존 v4 (보류 포함 모든 후보) | {len(verdicts)} | {old_counts['correct']} | {old_counts['incorrect']} |",
        f"| SVG 묶음 v5 | {len(verdicts)} | {counts['correct']} | {counts['incorrect']} |", '',
        '| 글자 | 전체 | 정답 | 오류 |', '|---|---:|---:|---:|',
    ]
    for char in '一二三四用':
        n = by_character[char]
        lines.append(f"| {char} | {sum(n.values())} | {n['correct']} | {n['incorrect']} |")
    lines += ['', f"기존 오류 개선 {result['improved']}건, 기존 정답의 회귀 {result['regressed']}건.", '',
              '## 별도 구성 사례', '',
              '각 글자의 첫 평가 필기에서 가장 긴 입력을 둘로 분리했다. 입력이 둘 이상이면 첫 두 입력을 연결선으로 이어 쓴 사례도 만들었다. 선택 기준은 v5 결과와 무관하다.',
              '실제 사용자의 새로운 끊어쓰기 자료가 아니라, 정답 대응을 알고 있는 원본을 변형한 동작 검사다.', '',
              '| 글자 | 원본 | 변형 | 결과 |', '|---|---|---|---|']
    for case in constructed:
        lines.append(f"| {case['character']} | {case['sample_id']} | {case['transformation']} | {'정답' if case['verdict']=='correct' else '오류'} |")
    metrics = result['relation_metrics']
    lines += ['', '## 범위와 제한', '',
              '정답은 구현에 참여한 AI가 시각적으로 검토한 개발 자료다. 독립 검수나 새 필기자에 대한 일반화 성능이 아니다. 원본의 미주석 19건은 정답으로 간주하지 않는다.',
              f'미주석 자료에서도 {len(missing_targets)}건은 SVG 획을 하나 이상 찾지 못했다. 따라서 전체 77건이 정확하다는 결과가 아니다.',
              '누락 사례: '+', '.join(f"{r['sample_id']} ({r['character']}, {r['missing']})" for r in missing_targets)+'.',
              '구간 경계 자체의 정확도는 평가하지 않았다. 완성된 필기만 이용하는 답안지 생성기이며 실시간 모델 학습은 실행하지 않았다.',
              f"연결 단위 TP {metrics['tp']}, FP {metrics['fp']}, FN {metrics['fn']}. 정밀도 {metrics['precision_pct']:.1f}%, 재현율 {metrics['recall_pct']:.1f}%. 글자 단위 정답 건수와 분모가 다르다.",
              '모든 생성 결과의 학습 사용 표시는 review_required다. 생성 완료는 정답 인증을 뜻하지 않는다.', '']
    (args.directory/'evaluation.md').write_text('\n'.join(lines))
    print(json.dumps({k:result[k] for k in ('current','baseline','improved','regressed','constructed_summary')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
