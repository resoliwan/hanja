"""Four fixed inference pipelines, identical plan evaluation images and sample IDs."""
import json,csv,hashlib,unicodedata,collections,itertools
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
from opencc import OpenCC
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/plan-fourway-evaluation';OLD=ROOT/'results/plan-threeway-evaluation';TRAIN=ROOT/'results/hanja3500-head-only-v1';DATA=ROOT/'data/korean-hanja3500-v1'
cc=OpenCC('s2t');norm=lambda c:unicodedata.normalize('NFC',c)
manifest=[json.loads(l) for l in (OLD/'evaluation-manifest.jsonl').read_text().splitlines()]
glyph=[json.loads(l) for l in (OLD/'glyph-scores.jsonl').read_text().splitlines()]
shikiji=[json.loads(l) for l in (OLD/'shikiji-scores.jsonl').read_text().splitlines()]
trained=list(csv.DictReader((TRAIN/'after-predictions.csv').open(encoding='utf-8-sig')))
assert len(manifest)==len(glyph)==len(shikiji)==len(trained)==73987
training_result=json.loads((TRAIN/'result.json').read_text())
checkpoint_hash=hashlib.sha256((TRAIN/'head-only-best.pt').read_bytes()).hexdigest();assert checkpoint_hash==training_result['best_checkpoint_sha256']
classes=(DATA/'classes.txt').read_text().splitlines();assert len(classes)==3500 and len(set(map(norm,classes)))==3500
old_classes=set((ROOT/'models/glyphweave-v1.0.0-inference/classes.txt').read_text().splitlines())
coverage=list(csv.DictReader((DATA/'character-coverage.csv').open(encoding='utf-8-sig')))
font_only={r['character'] for r in coverage if int(r['ai_free_images'])==0};assert len(font_only)==148
meta={norm(r['character']):{'grade':r['grade'],'known':r['character'] in old_classes,'font_only':r['character'] in font_only} for r in coverage}
assert sum(r['known'] for r in meta.values())==2115
acc=collections.defaultdict(lambda:collections.defaultdict(lambda:[0,0,0]))
models=['glyph','shikiji','ensemble','trained_glyph']
name={'glyph':'기존 GlyphWeave','shikiji':'Shikiji','ensemble':'기존 두 모델 점수순 결합','trained_glyph':'출력층 학습 GlyphWeave'}
changed=collections.Counter()

def merge(a,b):
 best={}
 for c in a+b:
  ch=c['character']
  if ch not in best or c['score']>best[ch]['score']:best[ch]=c
 return [c['character'] for c in sorted(best.values(),key=lambda c:(-c['score'],c['character']))[:5]]

with (OUT/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['sample_id','evaluation_set','label','grade','mode','model','candidates','top1','top5']);w.writeheader()
 for r,g,s,t in zip(manifest,glyph,shikiji,trained):
  sid=r['sample_id'];label=norm(r['character']);ev=r['evaluation_set']
  assert sid==g['id']==s['id']==t['sample_id'] and r['character']==g['label']==s['label']==t['character']
  rawt=t['candidates'].split();assert (rawt[0]==r['character'])==(t['top1']=='True');assert (r['character'] in rawt)==(t['top5']=='True')
  for mode in ['original','s2t']:
   conv=lambda c:norm(cc.convert(c) if mode=='s2t' else c)
   gc=[dict(character=conv(c['character']),score=c['score']) for c in g['candidates']]
   sc=[dict(character=conv(c['character']),score=c['score']) for c in s['candidates']]
   preds={'glyph':[c['character'] for c in gc],'shikiji':[c['character'] for c in sc],'ensemble':merge(gc,sc),'trained_glyph':[conv(c) for c in rawt]}
   if mode=='s2t':changed['trained_candidates_changed']+=sum(a!=b for a,b in zip(preds['trained_glyph'],map(norm,rawt)))
   for m,cs in preds.items():
    ok1=cs[0]==label;ok5=label in cs;a=acc[(mode,ev,m)][label];a[0]+=1;a[1]+=ok1;a[2]+=ok5
    w.writerow(dict(sample_id=sid,evaluation_set=ev,label=label,grade=r['grade'],mode=mode,model=m,candidates=' '.join(cs),top1=ok1,top5=ok5))

def stats(d):
 if not d:return dict(n=0,characters=0,top1=None,top5=None,macro_top1=None,macro_top5=None)
 n=sum(a[0] for a in d.values());a=sum(a[1] for a in d.values());b=sum(a[2] for a in d.values())
 return dict(n=n,characters=len(d),correct_top1=a,correct_top5=b,top1=a/n,top5=b/n,macro_top1=sum(v[1]/v[0] for v in d.values())/len(d),macro_top5=sum(v[2]/v[0] for v in d.values())/len(d))
summary={};grade_rows=[];char_rows=[]
for (mode,ev,m),d in acc.items():
 x=stats(d);x['by_grade']={grade:stats({c:v for c,v in d.items() if meta[c]['grade']==grade}) for grade in sorted(set(v['grade'] for v in meta.values()))}
 x['existing_characters']=stats({c:v for c,v in d.items() if meta[c]['known']})
 x['new_characters']=stats({c:v for c,v in d.items() if not meta[c]['known']})
 x['font_only_training_characters']=stats({c:v for c,v in d.items() if meta[c]['font_only']})
 summary.setdefault(mode,{}).setdefault(ev,{})[m]=x
 for grade,v in x['by_grade'].items():grade_rows.append(dict(mode=mode,evaluation_set=ev,model=m,grade=grade,**v))
 for c,v in d.items():char_rows.append(dict(mode=mode,evaluation_set=ev,model=m,character=c,grade=meta[c]['grade'],samples=v[0],correct_top1=v[1],correct_top5=v[2]))
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
for filename,rs in [('by-grade.csv',grade_rows),('by-character.csv',char_rows)]:
 with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=rs[0].keys());w.writeheader();w.writerows(rs)
# Reconcile reused trained results with the published training report, without trusting rounded values.
for ev,k in [('font_original','font_base'),('font_variants','font_variants'),('scut','scut_handwriting')]:
 for metric in ['top1','top5','macro_top1']:
  assert abs(summary['original'][ev]['trained_glyph'][metric]-training_result['after'][k][metric])<1e-10
validation=dict(samples_per_model=73987,matching_sample_ids_labels_order=True,classes=3500,trained_checkpoint_sha256=checkpoint_hash,trained_results_match_training_report=True,raw_image_hashes_checked_by_each_public_model_runner=True,source_provenance=json.loads((OLD/'provenance.json').read_text()),source_file_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OLD/'glyph-scores.jsonl',OLD/'shikiji-scores.jsonl',TRAIN/'after-predictions.csv']},**changed)
(OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
def pct(x):return '평가 불가' if x is None else str((Decimal(str(x))*100).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))+'%'
lines=['# 계획서 평가셋 · 네 방식 비교','', '**학습을 추가로 실행하지 않았습니다.** 기존 GlyphWeave, Shikiji, 기존 두 모델의 점수순 결합, 이미 학습한 GlyphWeave의 결과를 동일 표본으로 비교했습니다.','', '대상 목록은 계획서에 적힌 **simpledic 대한검정회 누적 3,500자**입니다. 이전 한국어문회 3,500자/첫 세트 11,999개 평가와 표본·목록이 다르므로 이전 퍼센트와 직접 전후 비교하면 안 됩니다.','', '## 모든 모델에 동일한 간체→번체 후처리 적용','', '아래 네 방식 모두 출력 후보에 OpenCC s2t와 Unicode NFC를 적용했습니다. 한국어문회/대한검정회 밖의 후보를 제거하지 않았습니다. 정답 글자 표기는 그대로입니다.','']
for ev,title in [('font_original','Noto Sans 원본 · 3,500장 / 3,500자'),('font_variants','Noto Sans 변형본 · 17,500장 / 3,500자'),('scut','SCUT 실제 필기 · 52,987장 / 3,072자')]:
 lines += ['### '+title,'','| 방식 | Top-1 | Top-5 | 문자별 평균 Top-1 |','|---|---:|---:|---:|']
 for m in models:
  x=summary['s2t'][ev][m];lines.append(f'| {name[m]} | {pct(x["top1"])} | {pct(x["top5"])} | {pct(x["macro_top1"])} |')
 lines+=['']
lines+=['## 학습 모델에는 번체 변환을 붙이지 않는 경우','', '학습 모델은 이미 한국 목표 3,500자 표기로 출력합니다. 따라서 범용 간체→번체 변환이 오히려 일부 한국식 표기를 바꿀 수 있습니다. 공통 후처리 비교와 구분하기 위해 원본 출력 결과도 제시합니다.','','| 평가 | 학습 모델 원본 Top-1 | 원본 Top-5 |','|---|---:|---:|']
for ev,title in [('font_original','Noto Sans 원본'),('font_variants','Noto Sans 변형본'),('scut','SCUT 필기')]:
 x=summary['original'][ev]['trained_glyph'];lines.append(f'| {title} | {pct(x["top1"])} | {pct(x["top5"])} |')
lines+=['','## 결합 규칙','', '기존 GlyphWeave와 Shikiji 각각 전체 출력 목록에 softmax를 적용한 상위 5개를 가져옵니다. 글자 변환 후 동일 후보는 더 높은 점수 하나만 유지하며, 점수 내림차순으로 최종 5개를 선택합니다. 학습된 GlyphWeave는 이 결합에 포함하지 않았습니다. 가중치·온도·임계값 튜닝이나 평가 정답을 이용한 후보 선정은 하지 않았습니다.','', '## 동일성 및 한계','', '- 공개 모델 두 개는 73,987개 평가 이미지 모두를 실제 읽고 픽셀 해시를 확인해 실행했습니다. 학습 모델은 이미 계산된 after-predictions.csv를 재사용했습니다. 전 모델의 평가 ID·정답·순서 73,987개가 일치하며, 학습 모델 파일 SHA-256이 학습 결과 기록과 일치함을 확인했습니다.','- 학습 모델의 원본 Top-1·Top-5·macro Top-1은 기존 학습 보고서와 세 평가셋 모두 일치합니다.','- 고정된 픽셀 해시 기준 중복을 제거했습니다. 폰트 원본/변형/필기 점수를 합산하지 않았습니다. SCUT에 없는 428자는 필기 평가 불가이며 0점으로 넣지 않았습니다.','- 기존 GlyphWeave는 원본 7,247자 모델입니다. 학습 계획서의 학습 전 모델은 3,500자로 출력층을 재구성하고 신규 행을 초기화한 모델이므로 이 비교의 기존 GlyphWeave와 다릅니다.','- 학습 후 모델은 목표 목록에 맞춰 학습되고 출력이 제한되어 있습니다. 이 표는 실제 사용 방식 비교이며 출력층 학습의 순수 효과만 분리하는 실험은 아닙니다.','- 기존 모델과 Shikiji의 학습 데이터 중복/필기자 독립성은 검증하지 않았습니다. 학습 실험에서 정확한 학습·평가 픽셀 중복 0건을 확인했다는 기록은 유사 글씨·필기자 중복이 없다는 뜻이 아닙니다.','- CPU/GPU와 실행 방식이 다르므로 속도 순위는 비교하지 않습니다.','', '## 상세 파일','', '- summary.json: 원본/s2t, 평가셋별, 급수별, 기존 2,115자/신규 1,385자, 폰트만 학습한 148자별 지표','- by-grade.csv: 급수별 Top-1·Top-5·문자별 평균·샘플 수','- by-character.csv: 문자별 정답 수','- predictions.csv: 네 방식의 원본 및 변환 후보 전체','- validation.json: 표본 동일성 및 모델·결과 파일 해시','', '재집계: `.venv/bin/python scripts/report_plan_fourway.py`']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:60]))
