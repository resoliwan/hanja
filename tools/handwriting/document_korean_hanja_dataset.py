"""Write the data card only after the build and complete-image verification."""
from paths import DATASETS
import csv,json
from pathlib import Path

OUT=DATASETS / 'korean-hanja3500-v1'

def main():
    s=json.loads((OUT/'summary.json').read_text());v=json.loads((OUT/'verification.json').read_text())
    fonts=json.loads((OUT/'font-coverage.json').read_text())
    rejected=json.loads((OUT/'excluded-aifree-filenames.json').read_text())
    blank=sum(r['reason']=='blank image' for r in rejected)
    rows=list(csv.DictReader((OUT/'character-coverage.csv').open(encoding='utf-8-sig')))
    text=f'''# 한국 한자 3,500자 이미지 데이터셋 v1

simpledic의 대한검정회 8급~1급 누적 한자 3,500자만 대상으로 만든 로컬 데이터셋입니다. 한글 음절, 영어, 숫자는 포함하지 않습니다. 모델 학습은 아직 실행하지 않았습니다.

## 생성 결과

| 출처 | 글자 수 | 이미지 수 | 종류 / 용도 |
|---|---:|---:|---|
| 폰트 5종 | {s['characters_by_source']['fonts']:,} | {s['counts_by_source']['fonts']:,} | 폰트 합성 / 학습 후보 |
| AI-FREE | {s['characters_by_source']['ai-free']:,} | {s['counts_by_source']['ai-free']:,} | 손글씨 유래 자료 / 학습 후보 |
| SCUT TradChar 5개 세트 | {s['characters_by_source']['scut']:,} | {s['counts_by_source']['scut']:,} | 실제 펜 궤적 렌더링 / 평가 전용 |
| 합계 | 3,500 | {v['verified_images']:,} | 출처별 구분 유지 |

- 손글씨 계열 출처(AI-FREE 또는 SCUT)가 있는 글자: {s['handwriting_source_characters']:,}자.
- 폰트 자료만 있고 손글씨 계열 자료가 없는 글자: {s['font_only_characters']:,}자.
- 어느 출처에도 이미지가 없는 글자: {s['no_image_characters']:,}자.
- AI-FREE 대상 이미지 중 빈 이미지 {blank:,}개를 제외했습니다. 파일 수는 실물 이미지 검증 전의 과거 집계와 다를 수 있습니다.

## 폰트 지원 범위

| 폰트 | 지원 | 누락 |
|---|---:|---:|
'''
    text+='\n'.join(f"| {f['font']} | {f['covered']:,} | {f['missing']:,} |" for f in fonts)
    text+='''

폰트마다 지원되는 문자만 그렸습니다. 다른 폰트로 자동 대체하지 않았고, Unicode 문자 변환도 하지 않았습니다. Noto CJK의 한국어판, 은글꼴 한국어 폰트, LXGW WenKai KR을 사용했습니다. 검토했던 Google Fonts 나눔고딕·나눔명조 파일에는 목표 한자가 없어 최종 5종에서 제외했습니다.

각 지원 폰트/글자 조합은 원본 1장과 회전 -3도/+3도, 좌우 이동, 크기 축소의 5개 변형으로 구성됩니다. 같은 원본에서 만든 6장은 같은 group_id입니다. 실제 필기자 6명의 자료로 취급하면 안 됩니다.

## 파일 구조

- `classes.txt`: class_id 순서의 정답 3,500자. 코드포인트 오름차순.
- `class-map.json`: 문자별 class_id와 최초 배정 급수.
- `character-coverage.csv`: 모든 글자의 폰트/AI-FREE/SCUT 이미지 수. 0도 명시.
- `shards/*.tar`: PNG 이미지 묶음. 각 PNG는 흰 배경·검은 글씨의 128×128 흑백 이미지.
- `manifests/fonts.csv`, `ai-free.csv`, `scut.csv`: 이미지별 정답, 출처, 원본 참조, 픽셀 해시, 용도, 묶음 파일 경로.
- `font-coverage.json`: 폰트별 누락 문자까지 기록.
- `font-only-characters.txt`: 현재 손글씨 계열 자료가 없는 문자.
- `no-images-characters.txt`: 어떤 이미지도 없는 문자. 없으면 빈 파일.
- `excluded-aifree-filenames.json`: 빈 이미지 및 잘못된 파일명 제외 기록.
- `verification.json`: 전체 이미지 검증 및 중복 집계.
- `training-excluded-pixel-hashes.txt`: 정답 충돌·평가 이미지와의 정확한 중복으로 학습에서 제외할 픽셀 해시.
- `source-documents/`: 폰트 이용 조건, 원본 자료 설명, 다운로드 주소·버전·해시.
- `SHA256SUMS.txt`: 최종 이미지 묶음 파일의 SHA-256.
- `preview.png`: 일부 문자별 5개 폰트와 두 손글씨 출처 비교.

## 자료의 의미와 한계

AI-FREE의 정확한 원본 필기자와 파일 생성 과정은 독립적으로 확인하지 않았습니다. 원본 README는 Tegaki 및 남대과학기술대학 자료를 출처로 명시합니다. 따라서 이 자료는 `handwriting_source_unverified`로 표시했습니다. 샘플을 눈으로 점검했지만 전체 이미지의 정답이나 사람 필기 여부를 수작업 검증했다는 뜻은 아닙니다.

SCUT는 현재 내려받은 TradChar 압축파일의 5개 세트 전체입니다. 각 POT 내부 문자 코드와 별도 정답을 모두 대조하고, 획 구분을 보존해 기존 평가와 같은 렌더러로 그렸습니다. 다섯 세트는 필기자 ID를 뜻하지 않습니다. 필기자 정보는 추측하지 않고 `unknown`으로 기록했습니다.

SCUT는 이전 모델 평가에 사용한 자료이므로 `evaluation_only`로 유지합니다. AI-FREE와 폰트만 `train_candidate`입니다. AI-FREE의 필기자 분리를 보장할 수 없어 검증셋/테스트셋을 임의로 만들지 않았습니다. 폰트 변형본을 무작위로 나누면 같은 원본이 학습·검증에 함께 들어갈 수 있습니다.

CASIA 및 Shikiji의 학습 이미지는 이번 묶음에 포함되지 않았습니다. 해당 모델의 지원 문자 목록이나 공개 샘플 수를 실제 확보한 이미지로 세지 않았습니다. 이체자 자동 변환이나 간체→번체 라벨 변경도 하지 않았습니다.

## 이용 조건

원본 이용 조건은 출처별로 유지됩니다. 전체 자료에 하나의 새 라이선스를 부여하지 않습니다.

- Noto 두 종과 LXGW WenKai KR: 첨부된 SIL Open Font License.
- 은바탕·은돋움: 첨부된 원본 COPYING(GPL) 및 다운로드 출처 참조.
- AI-FREE: 원본 README는 CC BY-NC-SA 4.0을 명시합니다. 상업적 사용을 승인받은 자료라고 간주하지 않습니다.
- SCUT IME_Test: 원본 README에 학술 연구용 공개 자료로 명시되어 있습니다.

## 불러오기

작업 폴더의 `tools/handwriting/hanja3500_dataset.py`에 이미지/정답 로더가 있습니다. 기본 학습 로더는 SCUT를 제외하고, 픽셀 단위 중복을 제거하며, 정답 충돌·평가와의 정확한 중복도 제외합니다.

```python
from scripts.hanja3500_dataset import Hanja3500Dataset
dataset = Hanja3500Dataset("resources/handwriting/datasets/korean-hanja3500-v1", usage="train")
image, class_id = dataset[0]
character = dataset.classes[class_id]
# 평가 자료는 별도로 선택:
evaluation = Hanja3500Dataset("resources/handwriting/datasets/korean-hanja3500-v1", usage="evaluation")
```

학습 시 GlyphWeave의 공식 transform을 image에 적용합니다. 폰트와 필기를 함께 쓰더라도 출처별 가중치와 문자별 샘플 균형을 별도로 조정해야 합니다.

## 재현

`tools/handwriting/prepare_dataset_sources.py` → `tools/handwriting/build_korean_hanja_dataset.py` → `tools/handwriting/verify_korean_hanja_dataset.py` → `tools/handwriting/document_korean_hanja_dataset.py` 순서입니다. 생성기는 이미 있는 최종 출력 폴더를 덮어쓰지 않습니다. 실제 사용된 원본 커밋과 다운로드 해시는 source-documents에 보존되어 있습니다.
'''
    (OUT/'README.md').write_text(text)
    data=json.dumps(rows,ensure_ascii=False).replace('<','\\u003c')
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>한국 한자 3,500자 자료</title>
<style>body{font:16px -apple-system,sans-serif;max-width:1200px;margin:36px auto;padding:0 20px;background:#f5f7fa;color:#172533}h1{font-size:30px}input,select{font:inherit;padding:10px;border:1px solid #ccc;border-radius:6px;margin:4px}table{border-collapse:collapse;width:100%;background:white}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:right}th{position:sticky;top:0;background:#dde7ee}td:first-child{font-size:27px;text-align:center}p{line-height:1.7}.bad{color:#a32929}.box{background:white;border-radius:10px;padding:20px}img{max-width:100%}</style>
<h1>한국 한자 3,500자 이미지 자료</h1>
<p>폰트 합성과 손글씨 계열 자료를 구분한 목록입니다. SCUT는 평가 전용입니다. AI-FREE의 원본 필기자와 합성 여부는 독립적으로 확인하지 않았습니다. 폰트 이미지가 있다고 실제 필기를 확보했다는 뜻은 아닙니다.</p>
<p><a href="README.md">데이터 설명</a> · <a href="character-coverage.csv">전체 CSV</a> · <a href="font-only-characters.txt">손글씨 자료 없는 글자</a></p>
<div class="box"><img src="preview.png" alt="폰트와 필기 자료 비교"></div>
<p><input id="query" placeholder="한자 또는 U+코드 검색"><select id="grade"><option value="">모든 급수</option></select><select id="kind"><option value="all">모든 글자</option><option value="fontonly">폰트 자료만 있음</option><option value="noai">AI-FREE 없음</option><option value="nosc">SCUT 없음</option></select><span id="count"></span></p>
<table><thead><tr><th>한자</th><th>코드</th><th>급수</th><th>폰트 수</th><th>폰트 이미지</th><th>AI-FREE</th><th>SCUT 평가</th><th>합계</th></tr></thead><tbody id="rows"></tbody></table>
<script>const data=DATA;
for(const g of ['8','7','6','준5','5','준4','4','준3','3','준2','2','준1','1']){const o=document.createElement('option');o.value=g;o.textContent=g+'급';document.querySelector('#grade').append(o)}
function update(){const q=document.querySelector('#query').value.trim().toUpperCase(),g=document.querySelector('#grade').value,k=document.querySelector('#kind').value;const rs=data.filter(r=>(!q||r.character.includes(q)||r.codepoint.includes(q))&&(!g||r.grade===g)&&(k==='all'||k==='fontonly'&&r.font_only==='True'||k==='noai'&&r.ai_free_images==='0'||k==='nosc'&&r.scut_evaluation_images==='0'));document.querySelector('#count').textContent=rs.length+'자';const body=document.querySelector('#rows');body.replaceChildren();for(const r of rs){const tr=document.createElement('tr');for(const key of ['character','codepoint','grade','font_count','font_images','ai_free_images','scut_evaluation_images','total_images']){const td=document.createElement('td');td.textContent=r[key];if(r[key]==='0')td.className='bad';tr.append(td)}body.append(tr)}}
for(const el of document.querySelectorAll('input,select'))el.addEventListener('input',update);update();</script></html>'''.replace('DATA',data)
    (OUT/'report.html').write_text(html)
    print('Wrote',OUT/'README.md')

if __name__=='__main__':main()
