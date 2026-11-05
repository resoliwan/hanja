"""Report separate font/handwriting accuracy and paired 3500-model comparisons."""
import collections
import csv
import json
from paths import DATASETS, RESULTS

RUN = RESULTS / 'hanja5000-head-only-v1'
BASE = RESULTS / 'hanja3500-head-only-v1'


def score(rows):
    grouped=collections.defaultdict(list)
    for row in rows:grouped[row['character']].append(row['top1']=='True')
    return dict(samples=len(rows),classes=len(grouped),
                top1=sum(r['top1']=='True' for r in rows)/len(rows) if rows else None,
                top5=sum(r['top5']=='True' for r in rows)/len(rows) if rows else None,
                macro_top1=sum(sum(v)/len(v) for v in grouped.values())/len(grouped) if grouped else None)


def main():
    result=json.loads((RUN/'result.json').read_text())
    assert result['status']=='complete'
    current=list(csv.DictReader((RUN/'after-predictions.csv').open(encoding='utf-8-sig')))
    old=list(csv.DictReader((BASE/'after-predictions.csv').open(encoding='utf-8-sig')))
    current_by_id={r['sample_id']:r for r in current}
    assert len(current_by_id)==len(current)
    assert all(r['sample_id'] in current_by_id and current_by_id[r['sample_id']]['character']==r['character'] for r in old)
    # Compare already-trained old model against expanded model on identical samples.
    paired={}
    groups=[('font_base','미학습 Noto Sans 원본',lambda r:r['source']=='fonts' and r['variant'].endswith(':base')),
            ('font_variants','미학습 Noto Sans 변형',lambda r:r['source']=='fonts' and not r['variant'].endswith(':base')),
            ('scut_handwriting','SCUT 실제 필기',lambda r:r['source']=='scut')]
    for key,_,select in groups:
        selected=[r for r in old if select(r)]
        paired[key]=dict(previous=score(selected),expanded=score([current_by_id[r['sample_id']] for r in selected]))
    coverage=list(csv.DictReader((DATASETS/'korean-hanja5000-v1/character-coverage.csv').open()))
    font_only_train={r['character'] for r in coverage if int(r['ai_free_images'])==0}
    missing_scut={r['character'] for r in coverage if int(r['scut_images'])==0}
    additional={}
    for name,chars in [('font_only_training',font_only_train),('without_scut_evaluation',missing_scut)]:
        additional[name]={key:score([r for r in current if select(r) and r['character'] in chars]) for key,_,select in groups}
    (RUN/'paired-baseline-comparison.json').write_text(json.dumps(dict(paired=paired,additional=additional),ensure_ascii=False,indent=2))
    lines=['# 사범 5,000항목(고유 4,998자) 확장 학습·평가','',
           '공식 목록의 汨 중복과 輻/輻 정준 동등 표기를 합친 4,998개 출력 클래스. 기존 3,500자 전부 유지, 1,498자 추가.','',
           f"기존 3,500자 최우수 모델에서 출력층을 확장했다. 특징 추출 부분은 고정했으며, AI-FREE 검증 macro Top-1 기준으로 {result['best_epoch']}회차를 선택했다. 총 {result['epochs']}회차 학습했다.",'',
           '## 최종 정확도','',
           '| 평가 | 문자 종류 | 이미지 수 | Top-1 | Top-5 | 문자별 평균 Top-1 |',
           '|---|---:|---:|---:|---:|---:|']
    for key,title,_ in groups:
        for sub,label in [(None,'전체'),('existing_characters','기존 3,500자'),('new_characters','추가 1,498자')]:
            m=result['after'][key] if sub is None else result['after'][key][sub]
            lines.append(f"| {title} / {label} | {m['classes']:,} | {m['n']:,} | {m['top1']:.2%} | {m['top5']:.2%} | {m['macro_top1']:.2%} |")
    lines+=['','## 기존 모델과 동일 이미지로 비교','',
            '| 평가 | 기존 3,500자 모델 Top-1 | 확장 모델 Top-1 | 변화 |',
            '|---|---:|---:|---:|']
    for key,title,_ in groups:
        b=paired[key]['previous']['top1'];a=paired[key]['expanded']['top1']
        lines.append(f'| {title} | {b:.2%} | {a:.2%} | {(a-b)*100:+.2f}%p |')
    lines+=['','## 검증 범위와 한계','',
            '- Top-1은 첫 후보 정답률, Top-5는 상위 5개 후보 안에 정답이 있는 비율이다. 문자별 평균은 이미지가 많은 문자의 영향을 줄인다.',
            '- SCUT 전체 5세트는 평가에만 사용했고, Noto Sans KR 전체도 평가 전용으로 남겼다. 학습·검증·평가 간 동일 픽셀 이미지 겹침은 0이다.',
            f'- SCUT 필기로 평가할 수 있는 문자는 4,043자다. 나머지 {len(missing_scut)}자는 실제 필기 정확도를 측정하지 못했다. 폰트 정확도와 필기 정확도는 합산하지 않는다.',
            f'- AI-FREE가 없는 {len(font_only_train)}자는 폰트만으로 학습했다. 두 원본에 모두 없는 274자 외에 SCUT에만 있는 2자도 평가 전용 정책 때문에 폰트만으로 학습했다.',
            '- AI-FREE 검증은 픽셀 해시 기준으로 분리했으며 필기자 독립성은 보장되지 않는다. 유사 이미지나 폰트 간 유사성까지 제거한 것은 아니다.',
            '- 폰트 생성 방식은 기존과 동일하다: 지원 글리프 확인, 128×128 흑백, 원본과 회전·이동·축소 6개 변형. 새 폰트 전용 글자 16개를 표본 시각 확인했다.',
            '- 고정 가중치 불변과 실제 저장 모델/특징 캐시의 Top-5 일치를 확인했다. 앱 모델 파일은 교체하지 않았다.',
            f"- 특징 캐시 {result['reused_features']:,}개 재사용, {result['new_features']:,}개 신규 계산. 학습 실행부터 최종 검증까지 {result['total_seconds']/60:.1f}분.",'',
            '모델: head-only-best.pt. 세부 결과: after-evaluation.json, after-per-character.csv, after-predictions.csv. 동일 샘플 비교: paired-baseline-comparison.json.','']
    (RUN/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(after={key:{k:v for k,v in result['after'][key].items() if k!='by_grade'} for key,_,_ in groups},paired=paired),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
