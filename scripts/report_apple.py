import json,csv,unicodedata,collections,statistics
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
from opencc import OpenCC
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/apple-set1'
r=[json.loads(l) for l in (OUT/'predictions.jsonl').read_text().splitlines()]
samples=json.loads((ROOT/'data/tradchar-set1.json').read_text())
assert len(r)==len(samples)==20000
assert [(x['id'],x['label']) for x in r]==[(x['id'],x['label']) for x in samples]
norm=lambda s:unicodedata.normalize('NFC',s)
chars=set((ROOT/'results/korean3500/inventory-3500.txt').read_text().strip())
cc=OpenCC('s2t');summary={}
for subset,rr in [('all',r),('korean3500',[x for x in r if norm(x['label']) in chars])]:
 out={}
 for mode in ['original','s2t']:
  preds=[[norm(cc.convert(c) if mode=='s2t' else c) for c in x['candidates']] for x in rr]
  out[mode]=dict(n=len(rr),unique_labels=len(set(norm(x['label']) for x in rr)),top1=sum(bool(c) and c[0]==norm(x['label']) for x,c in zip(rr,preds)),top5=sum(norm(x['label']) in c for x,c in zip(rr,preds)))
 out.update(errors=sum(x['error'] is not None for x in rr),no_detection=sum(not x['observations'] for x in rr),multiple_regions=sum(len(x['observations'])>1 for x in rr),candidate_count_histogram=dict(collections.Counter(len(x['candidates']) for x in rr)),median_ms=statistics.median(x['ms'] for x in rr))
 summary[subset]=out
summary['hak']=[x for x in r if x['label']=='學']
summary['config']=json.loads((OUT/'environment.json').read_text())
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
with (OUT/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
 keys=['id','label','candidates','error','top1','top5','ms','region_count'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({**{k:v for k,v in x.items() if k not in ['observations','candidates']},'candidates':' | '.join(x['candidates']),'region_count':len(x['observations'])} for x in r)
def pct(n,d):return str((Decimal(n)*100/Decimal(d)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
lines=['# Apple Vision OCR 비교','', 'macOS 내장 Vision `VNRecognizeTextRequest`를 실제 실행했습니다. Apple 키보드의 획 기반 필기 입력기/Scribble을 실행한 결과가 아닙니다.','', '## 전체 첫 세트 20,000개','', '| 조건 | Top-1 | Top-5 |','|---|---:|---:|']
for mode in ['original','s2t']:
 x=summary['all'][mode];lines.append(f'| {mode} | {pct(x["top1"],x["n"])}% ({x["top1"]:,}) | {pct(x["top5"],x["n"])}% ({x["top5"]:,}) |')
lines+=['',f'글자 검출 결과 없음: {summary["all"]["no_detection"]:,}개. 실행 오류: {summary["all"]["errors"]:,}개. 검출되지 않은 샘플도 오답으로 분모에 포함했습니다. 후보 수 분포: {summary["all"]["candidate_count_histogram"]}. Top-5를 요청해도 API가 후보 5개를 항상 주는 것은 아닙니다.','', '## 한국어문회 3,500자 해당 표본','', '기존과 동일한 3,158자 / 11,999개 필기에 대해 두 모델과 같은 OpenCC s2t 후처리를 적용했습니다. 나머지 342자는 필기 표본이 없어 평가하지 않았습니다.','', '| 모델 | Top-1 | Top-5 |','|---|---:|---:|']
other=json.loads((ROOT/'results/korean3500/summary.json').read_text())
for name,info in other['models'].items():
 x=info['s2t'];lines.append(f'| {name} | {pct(x["top1"],x["n"])}% | {pct(x["top5"],x["n"])}% |')
x=summary['korean3500']['s2t'];lines.append(f'| Apple Vision OCR | {pct(x["top1"],x["n"])}% | {pct(x["top5"],x["n"])}% |')
lines+=['','## 學 두 필기','']
for x in summary['hak']:lines.append(f'- {x["id"]}: '+(' / '.join(x['candidates']) if x['candidates'] else '검출/인식 결과 없음'))
lines+=['','## 조건과 제한','', '- macOS 26.5.1, Apple M4 Pro, Vision request revision 3, accurate, recognitionLanguages=[zh-Hant], usesLanguageCorrection=true, automaticallyDetectsLanguage=false. customWords나 정답 힌트는 주지 않았습니다.','- 같은 원본 필기 좌표와 기존 렌더러가 만든 128×128 PNG를 사용했습니다. 획을 추가하거나 지우지 않았습니다.','- 검출된 영역이 여러 개면 정답을 보지 않고 면적이 가장 큰 영역의 Top-5 문자열을 사용했습니다. 다른 영역은 JSONL observations에 보존했습니다.','- 문자열 전체가 정답과 일치해야 정답입니다. 여러 글자로 잘못 읽은 문자열에서 정답 한 글자만 골라내지 않았습니다. 앞뒤 공백만 제거했습니다.','- 처음 5,881개 이후 남은 샘플은 같은 설정의 네 프로세스로 나누어 처리했고 원래 순서대로 병합·검증했습니다. 실행 방식 차이로 처리시간은 속도 비교에 사용하지 않습니다.',
 '- 한국 한자 필터에는 앞선 보고서와 동일한 Unicode NFC를 적용했습니다. 원본/s2t는 별도로 집계했습니다.','- 보조 점검으로 앞 100개와 學 2개를 512×512로 확대했지만 결과가 더 나빴습니다(102개 중 8개 Top-1 정답; 128×128은 52개). 전체 평가는 처음 정한 128×128로 유지했습니다. 확대 테스트는 20,000개 전체 결과가 아닙니다.','- Vision은 이미지에서 글자 영역을 검출하고 읽는 OCR이며 단일 필기 분류기와 구조가 다릅니다. 이 결과가 애플 키보드 필기 입력기나 모든 Apple OCR 설정의 성능을 뜻하지 않습니다.','- 공식 API: https://developer.apple.com/documentation/vision/vnrecognizetextrequest','', '## 파일과 재실행','', '- predictions.jsonl: 모든 관측 영역·후보·정답·오류','- predictions.csv: 샘플별 선택 후보와 정오답','- summary.json: 전체/한국어문회 범위 통계','- environment.json: 실제 요청 revision·지원 언어·운영체제','', '`bin/apple_ocr data/apple-manifest.json data/apple-images results/apple-set1/predictions.jsonl`', '', '` .venv/bin/python scripts/report_apple.py `']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:30]))
