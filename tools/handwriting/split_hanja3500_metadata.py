"""Reserve a complete font for evaluation using CSV metadata only; no images read."""
from paths import DATASETS
import collections,csv,json
from pathlib import Path


OUT=DATASETS / 'korean-hanja3500-v1'
HELD_OUT='noto-sans-kr'

def main():
    path=OUT/'manifests/fonts.csv'
    with path.open() as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    for row in rows:
        row['usage']='evaluation_only' if row['variant'].split(':')[0]==HELD_OUT else 'train_candidate'
    temp=path.with_suffix('.tmp')
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temp.replace(path)
    perchar=collections.defaultdict(collections.Counter)
    counts=collections.Counter();labels=collections.defaultdict(set);eval_hashes=set();train_hashes=set()
    for file in sorted((OUT/'manifests').glob('*.csv')):
        with file.open() as f:
            for row in csv.DictReader(f):
                key=f"{row['source']}:{row['usage']}"
                counts[key]+=1;perchar[row['character']][key]+=1
                h=row['pixel_sha256'];labels[h].add(row['character'])
                (eval_hashes if row['usage']=='evaluation_only' else train_hashes).add(h)
    overlaps=eval_hashes&train_hashes
    conflicts={h for h,chars in labels.items() if len(chars)>1}
    excluded=overlaps|conflicts
    (OUT/'training-excluded-pixel-hashes.txt').write_text('\n'.join(sorted(excluded))+('\n' if excluded else ''))
    report=dict(held_out_font=HELD_OUT,held_out_characters=3500,counts=dict(counts),
        training_excluded_hashes=len(excluded),exact_train_evaluation_overlap_hashes=len(overlaps),
        conflicting_label_hashes=len(conflicts),audit='existing CSV labels and pixel hashes only; images not opened',
        variants_policy='All six variants per font/character stay in the same split')
    assert counts['fonts:evaluation_only']==21000
    assert counts['fonts:train_candidate']==82440
    (OUT/'split-summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    path=OUT/'character-coverage.csv'
    with path.open(encoding='utf-8-sig') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    fields=[x for x in fields if x not in ['font_training_images','font_evaluation_images']]+['font_training_images','font_evaluation_images']
    for row in rows:
        c=perchar[row['character']]
        row['font_training_images']=c['fonts:train_candidate']
        row['font_evaluation_images']=c['fonts:evaluation_only']
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    summary=json.loads((OUT/'summary.json').read_text());summary['split']=report
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (OUT/'SPLIT.md').write_text('''# 학습·평가 자료 구분

평가 폰트는 Noto Sans CJK KR입니다. 목표 3,500자를 모두 지원하는 두 폰트 중 하나입니다.

| 자료 | 학습 후보 | 평가 전용 |
|---|---:|---:|
| Noto Sans CJK KR | 0 | 21,000 |
| 나머지 폰트 4종 | 82,440 | 0 |
| AI-FREE | 174,971 | 0 |
| SCUT | 0 | 56,949 |

원본과 5개 변형을 모두 같은 용도로 배정했습니다. 이미지 묶음 파일을 다시 쓰지 않고 manifests의 usage를 변경했습니다. 따라서 데이터를 읽는 프로그램은 반드시 usage를 기준으로 선택해야 합니다. 제공 로더 tools/handwriting/hanja3500_dataset.py는 이 구분을 적용합니다.

학습 후보 총 257,411장, 평가 전용 총 77,949장입니다. 학습 로더는 기존에 기록된 픽셀 해시를 기준으로 평가 자료와의 정확한 중복, 정답 충돌, 학습 내부 중복을 추가로 제외하므로 실제 읽는 수는 더 적을 수 있습니다.

이미지 내용은 다시 읽지 않았습니다. 이번 검사는 기존 CSV 기록만 사용했으며 이미지 무결성 검사를 완료했다는 의미가 아닙니다. 모델 학습은 하지 않았습니다.
''')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
