from paths import RESULTS
import json,csv,unicodedata,collections,statistics
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
from opencc import OpenCC
OUT=RESULTS / 'ensemble-korean3500';cc=OpenCC('s2t')
norm=lambda s:unicodedata.normalize('NFC',s)
chars=set((RESULTS / 'korean3500/inventory-3500.txt').read_text().strip())
data={m:[json.loads(l) for l in (OUT/(m+'-scores.jsonl')).read_text().splitlines()] for m in ['glyph','shikiji']}
assert len(data['glyph'])==len(data['shikiji'])==11999
assert [(r['id'],r['label']) for r in data['glyph']]==[(r['id'],r['label']) for r in data['shikiji']]
validation={}
for m,path in [('glyph','tradchar-set1/glyph.jsonl'),('shikiji','shikiji-set1/predictions.jsonl')]:
 old={r['id']:r for r in map(json.loads,(RESULTS/path).read_text().splitlines())}
 differences=[r['id'] for r in data[m] if [c['character'] for c in r['candidates']]!=old[r['id']]['candidates']]
 validation[m+'_previous_candidate_mismatch_ids']=differences
summary={};output=[]
def merged(candidates):
 best={}
 for c in candidates:
  ch=c['character']
  if ch not in best or c['score']>best[ch]['score']:best[ch]=c
 return sorted(best.values(),key=lambda c:(-c['score'],c['character']))
for a,b in zip(data['glyph'],data['shikiji']):
 label=norm(a['label']);p={}
 for m,r in [('glyph',a),('shikiji',b)]:
  p[m]=[dict(character=norm(cc.convert(c['character'])),score=c['score'],model=m,original=c['character']) for c in r['candidates']]
 allc=merged(p['glyph']+p['shikiji']);ranked=allc[:5]
 # A secondary fixed policy: remove output candidates outside the 3500 inventory, without refilling from models.
 filtered=[c for c in allc if c['character'] in chars][:5]
 sets={'GlyphWeave':p['glyph'],'Shikiji':p['shikiji'],'max_score_merge':ranked,'max_score_merge_3500_filter':filtered,'union_up_to10':allc}
 row=dict(id=a['id'],label=label,glyph=p['glyph'],shikiji=p['shikiji'],merged=ranked,merged_filtered=filtered)
 row['metrics']={}
 for name,cs in sets.items():
  predictions=[c['character'] for c in cs];top1=bool(predictions) and predictions[0]==label;hit=label in predictions
  s=summary.setdefault(name,dict(n=0,top1=0,topk=0));s['n']+=1;s['top1']+=int(top1);s['topk']+=int(hit)
  row['metrics'][name]=dict(top1=top1,topk=hit)
 output.append(row)
summary['diagnostics']={
 'top1_model_selected':dict(collections.Counter(r['merged'][0]['model'] for r in output)),
 'top5_gained_vs_glyph':sum(r['metrics']['max_score_merge']['topk'] and not r['metrics']['GlyphWeave']['topk'] for r in output),
 'top5_lost_vs_glyph':sum(not r['metrics']['max_score_merge']['topk'] and r['metrics']['GlyphWeave']['topk'] for r in output),
 'top5_gained_vs_shikiji':sum(r['metrics']['max_score_merge']['topk'] and not r['metrics']['Shikiji']['topk'] for r in output),
 'top5_lost_vs_shikiji':sum(not r['metrics']['max_score_merge']['topk'] and r['metrics']['Shikiji']['topk'] for r in output),
 'union_correct_but_lost_by_ranking':sum(r['metrics']['union_up_to10']['topk'] and not r['metrics']['max_score_merge']['topk'] for r in output),
 'hak':[r for r in output if r['label']=='學']}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
(OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
(OUT/'merged.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output))
with (OUT/'merged.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['id','label','glyph','shikiji','merged','merged_top1','merged_top5']);w.writeheader()
 for r in output:w.writerow(dict(id=r['id'],label=r['label'],glyph=' '.join(f'{c["character"]}:{c["score"]:.6f}' for c in r['glyph']),shikiji=' '.join(f'{c["character"]}:{c["score"]:.6f}' for c in r['shikiji']),merged=' '.join(f'{c["character"]}:{c["score"]:.6f}({c["model"]})' for c in r['merged']),merged_top1=r['metrics']['max_score_merge']['top1'],merged_top5=r['metrics']['max_score_merge']['topk']))
def pct(n,d=11999):return str((Decimal(n)*100/Decimal(d)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
lines=['# GlyphWeave + Shikiji 원시 softmax 점수순 합치기','', '한국어문회 1급 누적 3,500자 중 기존 첫 세트에 표본이 있는 3,158자 / 11,999개 필기에서 두 모델을 다시 실행해 점수를 수집했습니다. 나머지 342자에 대한 인식 정확도는 평가하지 못했습니다.','', '| 방식 | Top-1 | Top-5 |','|---|---:|---:|']
for key,title in [('GlyphWeave','GlyphWeave 단독'),('Shikiji','Shikiji 단독'),('max_score_merge','두 모델 점수순 합치기'),('max_score_merge_3500_filter','합친 후보 중 한국어문회 외 글자 제외')]:
 s=summary[key];lines.append(f'| {title} | {pct(s["top1"])}% ({s["top1"]:,}) | {pct(s["topk"])}% ({s["topk"]:,}) |')
lines+=['','## 고정한 합치기 규칙','', '1. 두 모델 각각 전체 분류 목록에 대한 softmax를 계산하고 원래 상위 5개 후보와 점수를 가져옵니다. 상위 5개만 다시 정규화하지 않습니다.','2. 두 모델 모두 후보 글자에 OpenCC s2t와 Unicode NFC를 적용합니다.','3. 최대 10개 후보에서 같은 글자가 겹치면 더 높은 점수 하나를 남깁니다. 점수를 합산하거나 평균내지 않습니다.','4. 남은 후보를 점수 내림차순으로 정렬해 최종 상위 5개를 선택합니다. 같은 점수는 글자 순으로 정렬합니다.','5. 첫 합치기 결과는 한국어문회 외 후보도 그대로 포함합니다. 표의 마지막 행만 3,500자 밖의 후보를 제거한 보조 결과입니다. 어떤 조건에서도 각 모델의 원래 상위 5개 밖에서 후보를 추가하지 않습니다.','', '## 해석과 제한','', f'- 두 모델 원래 후보를 최대 10개까지 모두 유지하면 정답 포함률은 {pct(summary["union_up_to10"]["topk"])}%입니다. 이는 Top-5가 아니며, 이번 점수 정렬이 달성할 수 있는 후보 풀의 정답 포함 상한을 보여줍니다.',f'- 후보 풀에 정답이 있었지만 점수순 상위 5개에서 탈락한 샘플은 {summary["diagnostics"]["union_correct_but_lost_by_ranking"]:,}개입니다.', '- 두 모델의 softmax는 서로 보정되지 않은 점수입니다. 이번 실험은 원시 점수를 직접 비교하는 고정 규칙을 실제 평가한 결과이며, 점수가 실제 정답 확률이거나 두 모델에서 같은 의미라는 보장은 없습니다.','- 별도 가중치/온도/임계값을 평가 데이터에 맞춰 조정하지 않았습니다. 기존 결과로 이미 관찰한 세트이므로 독립적인 최종 검증셋 결과는 아닙니다.','- 원본 재실행 후보와 앞선 후보의 차이는 validation.json에 기록했습니다.','- 모델 버전·출처, 필기 데이터, 한국어문회 목록 및 Unicode 표기 처리의 한계는 ../korean3500/REPORT.md와 각 모델 보고서와 동일합니다.','', '## 學 사례','']
for r in summary['diagnostics']['hak']:lines.append(f'- {r["id"]}: '+', '.join(f'{c["character"]} {c["score"]*100:.2f}% ({c["model"]})' for c in r['merged']))
lines+=['','## 재실행','', '작업 폴더에서:', '```sh','.venv/bin/python tools/handwriting/ensemble_scores.py glyph','.venv/bin/python tools/handwriting/ensemble_scores.py shikiji','.venv/bin/python tools/handwriting/report_ensemble.py','```','', '원시 점수는 glyph-scores.jsonl / shikiji-scores.jsonl, 병합 후보는 merged.csv / merged.jsonl, 요약은 summary.json에 저장했습니다.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:12]));print(json.dumps({k:v for k,v in summary['diagnostics'].items() if k!='hak'},ensure_ascii=False));print(validation)
