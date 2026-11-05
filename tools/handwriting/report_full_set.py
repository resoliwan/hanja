import json,collections,csv
from decimal import Decimal, ROUND_HALF_UP
def fmt(value): return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
from pathlib import Path
from full_set import OUT,report
report()
summary=json.loads((OUT/'summary.json').read_text())
names={'hanzi':'Hanzi Lookup','zinnia':'Zinnia','cnn':'cnn_chinese_hw (이전 공개 모델)','glyph':'GlyphWeave'}
allrows={m:[json.loads(x) for x in (OUT/(m+'.jsonl')).read_text().splitlines()] for m in names}
coverage={m:json.loads((OUT/(m+'-coverage.json')).read_text()) for m in names}
missing={m:set(coverage[m]['missing_labels']) for m in names}
for m,rs in allrows.items():
    covered=[r for r in rs if r['label'] not in missing[m]]
    s=summary[m];s.update(top1_pct=100*s['top1']/s['n'],top5_pct=100*s['top5']/s['n'],covered_samples=len(covered),covered_top1_pct=100*sum(r['top1'] for r in covered)/len(covered),covered_top5_pct=100*sum(r['top5'] for r in covered)/len(covered))
    assert len(covered)==coverage[m]['covered_samples']
common=set.intersection(*[{r['label'] for r in rs if r['label'] not in missing[m]} for m,rs in allrows.items()])
common_metrics={m:dict(n=len(rr:=[r for r in rs if r['label'] in common]),top1=sum(r['top1'] for r in rr),top5=sum(r['top5'] for r in rr)) for m,rs in allrows.items()}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
(OUT/'common-vocabulary-summary.json').write_text(json.dumps(dict(unique_labels=len(common),models=common_metrics),indent=2))
# All 80,000 records must align, including failed samples.
expected=[r['id'] for r in allrows['hanzi']]
assert len(set(expected))==20000
assert all([r['id'] for r in rs]==expected for rs in allrows.values())
old=json.loads((OUT.parent/'tradchar-first100/predictions.json').read_text())
oldnames={'hanzi':'Hanzi Lookup','zinnia':'Zinnia','cnn':'cnn_chinese_hw (legacy)','glyph':'GlyphWeave'}
checks={m:sum(a['candidates']!=b['candidates'] for a,b in zip(rs[:100],[x for x in old if x['model']==oldnames[m]])) for m,rs in allrows.items()}
(OUT/'validation.json').write_text(json.dumps(dict(total_predictions=80000,unique_samples_per_model=20000,matching_sample_order=True,first100_candidate_mismatches=checks),indent=2))
lines=['# 번체 필기 한 세트 · 네 모델 비교','', '**SCUT-onHCCTestDB TradChar 첫 번째 세트: 20,000개 필기, 서로 다른 5,283자.** 원래 순서 그대로 모두 평가했으며 COUCH2009 결과는 아닙니다.','', '| 모델 | Top-1 | Top-5 | 오류 | 정답 글자 지원 범위 |','|---|---:|---:|---:|---:|']
for m,s in summary.items():lines.append(f'| {names[m]} | {fmt(s["top1_pct"])}% ({s["top1"]:,}/20,000) | {fmt(s["top5_pct"])}% ({s["top5"]:,}/20,000) | {s["errors"]:,} | {s["covered_samples"]:,}/20,000 |')
lines+=['','## 글자 지원 범위를 고려한 해석','', '위 표는 인식 불가능한 미지원 글자와 실행 오류까지 모두 포함한 실제 전체 정확도입니다. 지원하는 글자만 고르면 모델마다 평가 대상이 달라지므로 아래 수치를 전체 정확도와 혼동하면 안 됩니다.','', '| 모델 | 지원하는 정답만 평가한 Top-5 |','|---|---:|']
for m,s in summary.items():lines.append(f'| {names[m]} | {s["covered_top5_pct"]:.2f}% ({s["covered_samples"]:,}개 기준) |')
lines+=['', f'네 모델이 모두 지원하는 글자만 비교하면 {len(common):,}종, {common_metrics["hanzi"]["n"]:,}개 샘플입니다.','', '| 모델 | 공통 지원 글자 Top-1 | 공통 지원 글자 Top-5 |','|---|---:|---:|']
for m,s in common_metrics.items():lines.append(f'| {names[m]} | {100*s["top1"]/s["n"]:.2f}% | {100*s["top5"]/s["n"]:.2f}% |')
lines+=['','## 실행 조건과 확인','', '- 앞선 100자 테스트와 같은 공개 모델 및 전처리를 사용했습니다. 첫 100개에 대해 네 모델 모두 후보 목록이 이전 결과와 동일합니다.','- 모델/데이터 출처·해시·버전은 ../environment.json 및 ../REPORT.md에 있습니다. 원본은 https://github.com/HCIILAB/IME_Test 의 onHCCTestDB-TradChar.zip 입니다.','- 20,000개 모두 POT 내부 문자 코드와 별도 정답 파일을 대조했고, 획 경계 및 파일 끝까지 검증했습니다. 샘플을 제거하지 않았습니다.','- Top-1은 1순위 일치, Top-5는 5개 후보 내 포함입니다. 번체↔간체 변환이나 이체자 정규화는 하지 않았습니다.','- Hanzi Lookup 오류 328개는 모두 한 점짜리 획이 포함된 샘플이었습니다. 해당 샘플의 실패로 기록했고 WASM 상태를 다시 생성해 후속 샘플을 계속 평가했습니다.','- GlyphWeave는 처리 시간을 줄이기 위해 Apple M4 Pro의 MPS GPU에서 32개씩 처리했습니다. 나머지는 CPU입니다. 모델 정확도 비교이며, 이 실행의 처리시간으로 기기·모델 속도 순위를 매기면 안 됩니다.','- CNN은 사전학습 모델이 포함된 이전 TFLite 버전입니다. 최신 PyTorch 버전을 학습한 결과가 아닙니다.','- 첫 세트만 평가했습니다. 나머지 네 세트와 학습 데이터 중복/필기자 독립성은 검증하지 않았습니다.','', '## 파일 및 재실행','', '- predictions.csv: 네 모델의 80,000개 예측 전체','- 각 모델 .jsonl: 샘플별 후보·정답·오류 원시 기록','- *-coverage.json: 모델별 미지원 정답 목록','- summary.json: 전체/지원 글자별 정확도','- common-vocabulary-summary.json: 네 모델 공통 글자 정확도','- validation.json: 입력 순서·표본 수·기존 결과 일치 검증','', '작업 폴더에서 ` .venv/bin/python tools/handwriting/full_set.py extract `로 세트를 추출하고, ` .venv/bin/python tools/handwriting/full_set.py hanzi `의 마지막 인자를 `hanzi`, `zinnia`, `cnn`, `glyph`로 바꿔 각각 실행합니다. JSONL 결과가 있으면 완료된 기록 다음부터 재개합니다. 최종 보고서는 ` .venv/bin/python tools/handwriting/report_full_set.py `로 생성합니다.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print((OUT/'REPORT.md').read_text())
