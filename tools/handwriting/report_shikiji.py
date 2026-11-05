from paths import DATASETS, RESULTS
import json,csv
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
from opencc import OpenCC
p=RESULTS / 'shikiji-set1'
r=[json.loads(l) for l in (p/'predictions.jsonl').read_text().splitlines()]
samples=json.loads((DATASETS / 'tradchar-set1.json').read_text())
assert len(r)==len(samples)==20000 and [x['id'] for x in r]==[x['id'] for x in samples]
assert all(a['label']==b['label'] for a,b in zip(r,samples))
s=json.loads((p/'summary.json').read_text())
old=json.loads((RESULTS / 'tradchar-set1/summary.json').read_text())
def pct(n,d=20000):return str((Decimal(n)*100/Decimal(d)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
conv=OpenCC('s2t')
converted=[dict(x,converted_candidates=[conv.convert(c) for c in x['candidates']]) for x in r]
converted_summary=dict(method='OpenCC s2t applied only to output candidates; ground-truth unchanged; fixed original top-5 with no extra candidates',n=20000,top1=sum(x['converted_candidates'][0]==x['label'] for x in converted),top5=sum(x['label'] in x['converted_candidates'] for x in converted))
(p/'s2t-summary.json').write_text(json.dumps(converted_summary,ensure_ascii=False,indent=2))
with (p/'predictions-s2t.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['id','label','candidates','converted_candidates']);w.writeheader();w.writerows(dict(id=x['id'],label=x['label'],candidates=' '.join(x['candidates']),converted_candidates=' '.join(x['converted_candidates'])) for x in converted)
lines=['# Shikiji ONNX 실행 결과','', '**SCUT-onHCCTestDB TradChar 첫 세트: 20,000개 샘플, 서로 다른 5,283자.** 앞선 네 모델과 동일한 샘플·정답 순서입니다.','', '| 모델 | Top-1 | Top-5 |','|---|---:|---:|']
for m,n in [('hanzi','Hanzi Lookup'),('zinnia','Zinnia'),('cnn','cnn_chinese_hw (이전 공개 모델)'),('glyph','GlyphWeave')]:
 x=old[m];lines.append(f'| {n} | {pct(x["top1"])}% | {pct(x["top5"])}% |')
lines.append(f'| **Shikiji ONNX v0.0.2** | **{pct(s["top1"])}%** | **{pct(s["top5"])}%** |')
lines+=['',f'정확히 일치한 샘플 수: Top-1 {s["top1"]:,}개, Top-5 {s["top5"]:,}개. 실행 오류 {s["errors"]}개. 정답 글자 지원 범위는 {s["covered_samples"]:,}/20,000개이며, 이번 세트에서 미지원 정답은 {len(s["missing_labels"]):,}종입니다.','',f'지원하는 정답 글자만 평가하면 Top-5 {pct(s["top5"],s["covered_samples"])}%입니다. 평가 대상이 달라지므로 전체 정확도와 구분해야 합니다.','', '## 배울 학 學 두 사례','', 'Shikiji의 분류 목록에는 **學과 学 모두** 있습니다. 하지만 이번 두 필기 모두 간체 **学**을 1순위로 출력했습니다.','', '| 샘플 | 정답 | 원래 Top-5 후보 |','|---|---|---|']
for x in r:
 if x['label']=='學':lines.append(f'| {x["id"]} | 學 | {" ".join(x["candidates"])} |')
lines+=['','## 간체→번체 후처리 별도 실험','',f'원래 후보 5개 각각에 OpenCC `s2t` 변환을 적용하면 Top-1 **{pct(converted_summary["top1"])}%**, Top-5 **{pct(converted_summary["top5"])}%**입니다. 정답은 변환하지 않았고 후보를 추가하지도 않았습니다. 위의 원본 모델 비교 표와는 별도 결과입니다.','', '學 두 사례는 출력 후보의 学을 學으로 변환하면 모두 1순위 정답이 됩니다. 다만 간체·번체가 항상 일대일 대응하는 것은 아니므로 이런 변환이 모든 글자를 해결하지는 않습니다.','', '## 실행 방법과 검증','', '- 공개 모델: https://huggingface.co/kwadraten/shikiji, v0.0.2. 저장소 revision 및 공식 LFS SHA-256 대조 결과는 provenance.json에 보관했습니다.','- 모델 파일 143,951,632 bytes, 분류 대상 10,596종. ONNX Runtime CPUExecutionProvider, CPU 8스레드, batch 32로 실행했습니다.','- 이전 테스트와 동일한 원본 획 좌표를 동일한 렌더러로 128×128에 그렸습니다. 이후 공식 메타데이터에 맞게 160×160 RGB로 리사이즈하고 float32 [0,1] NCHW로 입력했습니다. 추가 평균/표준편차 정규화는 하지 않았습니다.','- 원본 비교는 정확한 Unicode 문자 일치이며 간체/번체·일본 신자체를 같은 글자로 취급하지 않았습니다. 분류 목록 전체에서 Top-5를 구했습니다.','- 학습 자료와 평가 자료의 중복 여부는 검증하지 않았습니다. 공식 모델 카드의 검증 정확도와는 데이터가 다릅니다.','- 시간은 전처리 포함 배치 처리량을 샘플당 환산한 값입니다. 이전 모델의 서로 다른 CPU/GPU 실행시간과 직접 속도 순위를 매기지 않습니다.','', '## 파일','', '- predictions.csv / predictions.jsonl: 전체 원본 후보','- predictions-s2t.csv / s2t-summary.json: 출력만 간체→번체 변환한 별도 결과','- summary.json: 정확도·지원 범위·미지원 글자','- provenance.json: 모델 revision, 크기, SHA-256','', '재실행: 작업 폴더에서 `.venv/bin/python tools/handwriting/run_shikiji.py`, 이어서 `.venv/bin/python tools/handwriting/report_shikiji.py`. 기존 JSONL 파일이 있으면 처리한 샘플 다음부터 재개합니다.']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n')
print((p/'REPORT.md').read_text())
