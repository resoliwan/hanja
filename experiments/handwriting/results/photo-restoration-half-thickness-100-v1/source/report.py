"""Summarize the pilot without treating synthetic variants as independent photos."""

import argparse
import json
from pathlib import Path

import numpy as np

from paths import RESULTS
from .evaluate import summarize


def _paired_interval(rows, baseline='auto-binary'):
    # Resample originals, retaining all conditions generated from each original.
    groups = {}
    for row in rows:
        delta = int(row['methods']['restored']['rank'] == 1) - int(row['methods'][baseline]['rank'] == 1)
        groups.setdefault(row['sample_id'], []).append(delta)
    values = np.array([np.mean(v) for v in groups.values()])
    rng = np.random.default_rng(20261104)
    estimates = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(2000)])
    return dict(originals=len(values), difference=float(values.mean()), bootstrap_95_interval=np.quantile(estimates, [.025, .975]).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=RESULTS / 'photo-restoration-pilot-v1')
    output = parser.parse_args().output
    test = json.loads((output / 'test-results.json').read_text())
    result = json.loads((output / 'result.json').read_text())
    history = json.loads((output / 'history.json').read_text())
    known = json.loads((output / 'known-photo-results.json').read_text())
    stress = [r for r in test['rows'] if r['condition'] != 'clean']
    analysis = dict(stress_summary=summarize(stress)['all'], paired_gain=_paired_interval(stress),
        by_source={source: summarize([r for r in stress if r['source'] == source])['all'] for source in sorted({r['source'] for r in stress})})
    analysis['known_photo_variants'] = summarize([r for r in known['rows'] if r['condition'] != 'original'])['all']
    analysis['known_originals'] = summarize([r for r in known['rows'] if r['condition'] == 'original'])['all']
    (output / 'analysis.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2))
    lines = ['# 사진 획 복원 파일럿 결과', '',
        '**판정: 합성 데이터에서 개선은 확인했지만, 기존 사진의 색상 변형에서 크게 실패해 현재 모델의 앱 적용은 보류한다.**', '',
        '500자, 원본 20,000개 슬롯, 3 epoch. 매 epoch 1,000개 슬롯은 빈 배경으로 대체되어 실제 글자 원본 19,000개를 사용했다. 기존 한자 인식기는 고정했다.', '',
        f"선택된 체크포인트: epoch {result['best_epoch']} (검증 Top-1, 동률 시 Dice). 준비·학습·검증·최종 평가 포함 {result['elapsed_seconds'] / 60:.1f}분.", '',
        '최종 평가는 학습·검증과 원본 그룹 및 픽셀 해시가 겹치지 않는 1,000개 원본(500자)을 사용했다. 원본당 6조건, 총 6,000장. 같은 합성기에서 다른 seed로 만든 데이터이므로 실제 사진 일반화 성능을 뜻하지 않는다.', '',
        '## 조건별 정답 1순위', '',
        '|조건|장수|깨끗한 정답 입력|회색조|기존 흑백 방식|자동 반전 흑백|학습 복원|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for condition, methods in test['summary'].items():
        values = [f"{methods[m]['top1'] * 100:.1f}%" for m in ['clean-target', 'gray-only', 'current-binary', 'auto-binary', 'restored']]
        lines.append(f"|{condition}|{methods['restored']['n']}|" + '|'.join(values) + '|')
    lines += ['', '## 색·배경·조명 변형만 비교 (clean 제외)', '', '|처리|Top-1|Top-10|', '|---|---:|---:|']
    for method, metrics in analysis['stress_summary'].items():
        lines.append(f"|{method}|{metrics['top1'] * 100:.1f}%|{metrics['top10'] * 100:.1f}%|")
    gain = analysis['paired_gain']
    lo, hi = gain['bootstrap_95_interval']
    lines += ['', f"자동 반전 흑백 대비 Top-1 차이: {gain['difference'] * 100:+.1f}%p. 원본 단위 paired bootstrap 95% 구간 [{lo * 100:+.1f}, {hi * 100:+.1f}]%p. 합성기나 실제 촬영 환경의 불확실성까지 반영하는 구간은 아니다.", '',
        '## 학습 경과', '', '|epoch|loss|검증 복원 Top-1|검증 Dice|시간(분)|', '|---|---:|---:|---:|---:|']
    for row in history:
        lines.append(f"|{row['epoch']}|{row['train_loss']:.4f}|{row['validation']['restored']['top1'] * 100:.1f}%|{row['validation_dice']:.3f}|{row['seconds'] / 60:.1f}|")
    blank_path = output / 'blank-results.json'
    if blank_path.exists():
        blank = json.loads(blank_path.read_text())
        lines += ['', '## 글자 없는 배경 진단', '',
            f"별도 seed로 만든 빈 배경 {blank['n']}장 중 {sum(r['foreground_pixels'] > 0 for r in blank['rows'])}장에서 foreground probability ≥ 0.5인 픽셀이 발생했다. 16픽셀 이상 나온 경우는 {sum(r['foreground_pixels'] >= 16 for r in blank['rows'])}장이다. 실제 사진에서 보정한 거부 기준이 아니며 배경 무늬에서 오검출이 남아 있다."]
    lines += ['', '## 기존 사진 및 색상 변형 — 중요한 실패', '',
        '|진단 대상|장수|자동 반전 Top-1|복원 Top-1|복원 Top-10|', '|---|---:|---:|---:|---:|']
    for label, key in [('사진 원본 超·譯', 'known_originals'), ('같은 사진의 색·배경·가운데 조명 변형', 'known_photo_variants')]:
        metrics = analysis[key]
        lines.append(f"|{label}|{metrics['restored']['n']}|{metrics['auto-binary']['top1'] * 100:.1f}%|{metrics['restored']['top1'] * 100:.1f}%|{metrics['restored']['top10'] * 100:.1f}%|")
    lines += ['',
        '이 두 글자 사진은 복원 학습에는 쓰지 않았지만 이전부터 반복 관찰한 진단 자료이므로 독립 실사 정확도를 추정하는 자료가 아니다. 32개 변형의 자동 반전 결과가 과거 23/32와 달리 이번에는 24/32인 것은 256px 리사이즈 및 Python 전처리 경로가 다른 비교이기 때문이다. 위 표는 이번의 동일한 입력·평가 경로에서 비교했다.',
        '관찰: 실제 사진 계열의 복원 결과에는 획 내부가 비거나 테두리만 남고, 얼룩이 발생한다. 특히 譯의 색상 변형에서 실패가 많다. 원인은 아직 확정하지 않았으며, 학습 이미지와 굵은 인쇄 글씨·실제 질감의 분포 차이가 후보이다.',
        '다음 실험은 클래스 수만 늘리기보다 굵은 획·글자 크기·전경 내부 질감·실제 배경을 보강하고, 실제 사진 그룹으로 분리된 검증/테스트 세트를 확보하는 데 우선순위를 둔다. 현재 실패 사진은 개발용 회귀 사례로 유지하고 새 실제 사진을 최종 평가용으로 별도 확보한다. 이 파일럿 이후 추가 학습·배포는 수행하지 않았다.', '',
        '![원본 사진과 복원 결과](original-photo-restoration.png)', '',
        '![기존 사진 변형 실패 사례](known-photo-examples.png)']
    lines += ['', '## 재현 및 한계', '',
        '- 실행: `PYTHONPATH=tools/handwriting .venv/bin/python -m photo_restoration.train`. 기존 체크포인트가 있으면 덮어쓰지 않고 중단한다. 새 실행은 --output으로 경로를 지정한다.',
        '- 입력 RGB 256×256, 출력 soft mask, 최종 foreground cutoff 0.5는 test 이전 고정. 약 110만 파라미터, M4 Pro MPS FP32.',
        '- 평가용으로 지정된 폰트 원본과 SCUT은 학습에서 제외했다. 각 클래스에서 가능하면 글꼴 1개와 SCUT 손글씨 1개를 평가했다. 별도 by_source 결과는 analysis.json에 포함한다.',
        '- 학습과 검증의 원본 및 seed는 다르지만 배경 생성 규칙은 공유한다. 실제 재질 사진, JPEG, 원근·주변 글자 처리는 이번 파일럿에 포함하지 않았다.',
        '- 후보 분류는 현재 앱 ONNX를 사용했다. 전처리는 Python/Pillow로 구현한 비교 경로이며 모바일 JPEG 변환·리사이즈와 완전히 동일한 실행 경로는 아니다. 앱 적용 전 Dart/ONNX parity 검증이 필요하다.',
        '- 기존 2글자 사진과 32색상 변형은 known-photo-results.json의 별도 진단이며 독립적인 실사 평가가 아니다.',
        '- 빈 배경 학습을 포함하지만 제품용 거부 임계값과 실제 사진 정확도는 아직 검증되지 않았다.',
        '- 설정·원본 split·manifest hash·학습 이력·체크포인트·전체 예측을 같은 폴더에 보관했다. 앱 모델과 필기 경로는 변경하지 않았다.', '',
        '![합성 입력과 정답](training-pairs.png)', '', '![평가 입력, 정답, 복원 결과 순서](test-examples.png)']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
