# 대한검정회 사범 범위 확장 데이터셋

공식 5,000개 항목의 중복 및 유니코드 정준 동등 표기를 합친 4,998개 클래스입니다.
기존 3,500자 이미지는 바꾸지 않고 새 1,498자 이미지만 생성했습니다.
원본 출처 목록 및 비교 결과는 experiments/handwriting/results/sabeom5000-coverage에 있습니다.

- AI-FREE: 246,466장 / 4,722자 (학습·검증 후보)
- SCUT: 75,048장 / 4,043자 (평가 전용)
- 폰트 5종: 143,874장 / 4,998자
- Noto Sans KR는 전부 평가 전용. 다른 4종 폰트는 학습 후보.
- 이미지 크기 128×128, 기존과 동일한 원본·회전·이동·축소 변형.
- 정규화: Unicode NFC만 적용. 간체·번체 변환이나 임의 이체자 치환 없음.

실제 로더는 동일 픽셀 중복을 제외합니다. 학습 317,494장, 검증 19,576장,
최종 평가 99,807장입니다. validation-sample-ids.txt로 검증을 분리합니다.
기존 검증 ID는 유지하고, 신규 AI-FREE 문자는 픽셀 해시 순으로 약 8%를 검증에 남겼습니다.
필기자 ID가 없어 필기자 독립 검증을 보장하지 않습니다.

training-excluded-pixel-hashes.txt에는 모든 평가 픽셀 해시 및 라벨 충돌 해시가
기록됩니다. 이번 라벨 충돌은 0개입니다. 학습 로더가 해당 해시를 제외합니다.
base-shards는 기존 3,500자 tar의 하드링크이며, 원본과 함께 불변 파일로 취급합니다.
새 이미지 묶음은 shards에 있습니다. manifests의 shard 경로로 읽어야 합니다.

폰트만으로 학습하는 문자는 276자입니다. 두 필기 원본에 모두 없는 274자와
SCUT에만 있는 2자가 해당합니다. SCUT를 학습에 넣지 않습니다.

재현 도구:
- tools/handwriting/check_sabeom_coverage.py
- tools/handwriting/expand_sabeom_dataset.py (새 출력 폴더에만 실행)
- tools/handwriting/train_sabeom_head.py
- tools/handwriting/report_sabeom_evaluation.py

AI-FREE/SCUT 및 폰트의 기존 출처·이용 조건은 그대로 적용됩니다.
