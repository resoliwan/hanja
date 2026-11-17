"""Generate ten cases per supplied SVG and run both actual implementations."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from paths import ROOT, RESULTS, TOOLS
from stroke_mapping.synthetic_cases import generate_svg_cases
from stroke_mapping.target_matcher import match_targets
from stroke_mapping.sources import read_grade_entries


def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def relations(row):
    return {(m['source_stroke_number'],m['svg_stroke_number']) for m in row['mappings']}


def metrics(cases):
    tp=fp=fn=exact=paint=0
    for expected,predicted in cases:
        tp+=len(expected&predicted); fp+=len(predicted-expected); fn+=len(expected-predicted)
        exact+=expected==predicted
        paint+={j for _,j in expected}=={j for _,j in predicted}
    return dict(total=len(cases),correct=exact,incorrect=len(cases)-exact,paint_correct=paint,
                tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--characters',default='一二三四用')
    p.add_argument('--grade',help='Simpledic 대한검정회 신출 급수')
    p.add_argument('--svg',type=Path,help='Generate ten cases for this one SVG instead of a character list')
    p.add_argument('--out',type=Path,default=RESULTS/'stroke-matcher-comparison-5-v1')
    p.add_argument('--svg-dir',type=Path,default=ROOT.parent/'simpledic/data/dictionaries/hanja-ko/assets/stroke-order')
    p.add_argument('--shisen',type=Path,default=ROOT.parent/'shisen/apps/game')
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    if args.grade:
        if args.svg:
            p.error('--grade and --svg are mutually exclusive')
        entries=read_grade_entries(args.svg_dir.parent.parent,args.grade)
        args.characters=''.join(e['character'] for e in entries)
    paths=[args.svg] if args.svg else [args.svg_dir/f'u{ord(c):04x}.svg' for c in args.characters]
    cases=[]
    for path in paths:
        generated=generate_svg_cases(path)
        assert generated==generate_svg_cases(path),'Generator is not deterministic'
        cases.extend(generated)
    dataset=args.out/'cases.jsonl'
    dataset.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases))
    source_files=['dm/model/handwriting.dart','dm/model/shooter_glyph.dart',
                  'dm/model/shooter_stroke_evaluation.dart','dm/service/shooter_stroke_matcher.dart',
                  'dm/usecase/evaluate_writing_stroke.dart']
    provenance={}
    for relative in source_files:
        source=args.shisen/'lib'/relative
        target=args.out/'shisen'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        provenance[relative]=hashlib.sha256(source.read_bytes()).hexdigest()
    runner=args.out/'run_shisen.dart'
    shutil.copyfile(TOOLS/'shisen_stroke_runner.dart',runner)
    subprocess.run(['dart',str(runner),str(dataset),str(args.out/'shisen-results.jsonl')],check=True)
    shisen={r['case_id']:r for r in map(json.loads,(args.out/'shisen-results.jsonl').read_text().splitlines())}
    evaluated=[]; target_rows=[]
    for case in cases:
        start=time.perf_counter()
        result=match_targets(case['character'],case['strokes'],case['template'])
        duration=time.perf_counter()-start
        target_rows.append(dict(case_id=case['case_id'],runtime_seconds=duration,result=result))
        expected=set(map(tuple,case['expected_relations']))
        predictions={'target':relations(result),'shisen':set(map(tuple,shisen[case['case_id']]['relations']))}
        evaluated.append(dict(case_id=case['case_id'],character=case['character'],variant=case['variant'],
                              policy=case['evaluation_policy'],expected=sorted(expected),
                              predictions={k:sorted(v) for k,v in predictions.items()}))
        if len(evaluated)%50==0:
            print(f'Compared {len(evaluated)}/{len(cases)} cases',flush=True)
    (args.out/'target-results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in target_rows))
    score={}
    for algorithm in ('shisen','target'):
        score[algorithm]={}
        for variant in ['all','correspondence_only']+[c['variant'] for c in cases[:10]]:
            selected=[r for r in evaluated if variant=='all' or
                      (variant=='correspondence_only' and r['policy']=='segment_correspondence') or r['variant']==variant]
            score[algorithm][variant]=metrics([(set(map(tuple,r['expected'])),set(map(tuple,r['predictions'][algorithm]))) for r in selected])
    summary=dict(records=len(cases),scores=score,
                 source_snapshot=provenance,dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                 local_source_hashes={name:hashlib.sha256((TOOLS/'stroke_mapping'/name).read_bytes()).hexdigest()
                                      for name in ('synthetic_cases.py','target_matcher.py','geometry.py','guided_matcher.py','segment_matcher.py')},
                 timings_seconds=dict(target=sum(r['runtime_seconds'] for r in target_rows),
                                      shisen=sum(r['runtime_seconds'] for r in shisen.values())),
                 notes=['Synthetic SVG-derived tests, not independent human handwriting.',
                        'Different-character cases use explicit whole-character rejection policy; shared strokes may legitimately match in a partial-coloring product. Report these separately.',
                        'Join on a one-stroke glyph is a split/reconnect identity case.',
                        'Shisen is its actual sequential use case, target is offline full-glyph assignment. Timing is total per case, not equal runtime or real-time latency.',
                        'Metrics compare input-to-SVG relations and final colored SVG sets. Interval boundary accuracy is not scored in this pilot.'])
    write_json(args.out/'summary.json',summary); write_json(args.out/'evaluated.json',evaluated)
    lines=['# 두 획 매칭 알고리즘 비교','',f'SVG {len(paths)}개에서 실행 시 결정적으로 생성한 {len(cases)}건. 같은 SVG 바이트와 생성기 버전이면 같은 데이터가 나온다.','',
           '| 변형 | 건수 | 사천성 정답 | SVG 묶음 정답 |','|---|---:|---:|---:|']
    for variant in score['target']:
        a=score['shisen'][variant];b=score['target'][variant]
        lines.append(f"| {variant} | {b['total']} | {a['correct']} | {b['correct']} |")
    lines+=['','정답은 입력→SVG 대응이 모두 맞는 경우다. 최종 SVG 색칠 집합의 정답 수는 summary.json의 paint_correct에 별도로 기록했다.',
            f'다른 글자 {len(paths)}건은 전체 글자가 다르면 아무 획도 색칠하지 않는 별도 정책 검사다. 공통 획 색칠을 허용하는 제품에서는 이 검사를 주 정확도에 섞으면 안 된다. correspondence_only가 이를 제외한 주 비교다.',
            '원본에 점을 가미한 합성 필기이며 실제 사람 필기의 정확도가 아니다. 두 알고리즘의 내부 점수를 직접 비교하지 않았다.',
            '사천성은 실제 EvaluateWritingStroke와 ShooterStrokeMatcher를 바이트 그대로 복사해 Dart로 실행했다. 앱 소스는 수정하지 않았다. 내부 입력 번호가 거절 입력을 제외하는 부분만 어댑터에서 원래 입력 번호로 복원했다.',
            '사천성은 모든 입력을 순차 처리한 최종 결과, SVG 묶음은 완성된 글자 한 번의 결과다. 실시간 중간 단계 성능은 이번 비교 대상이 아니다.',
            '一처럼 한 획인 글자의 이어쓰기 검사는 같은 획을 나눴다 다시 잇는 항등 사례다. 구간 경계 자체의 정밀도·재현율은 이번 집계에 포함하지 않았다.','']
    (args.out/'report.md').write_text('\n'.join(lines))
    print(json.dumps({k:{v:s[v] for v in ('all','correspondence_only')} for k,s in score.items()},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
