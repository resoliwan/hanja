# hanja

한국 한자 3,500자 필기 인식 실험, 학습 데이터, GlyphWeave 추가 학습 모델 및 Flutter 앱입니다.

## 시작하기

Git LFS를 설치한 환경에서 내려받습니다. 데이터와 모델이 포함되어 있어 다운로드 용량은 수 GB입니다.

```sh
git lfs install
git clone --recurse-submodules https://github.com/resoliwan/hanja.git
cd hanja
git lfs pull
```

이미 복제한 경우 `git submodule update --init --recursive`로 외부 프로젝트를 가져옵니다. 외부 프로젝트는 수정하지 않은 원본 커밋을 참조합니다.

Flutter 앱:

```sh
cd hanja_app
flutter pub get
flutter run
```

Python 실험은 별도 가상환경을 만들고 [기록된 패키지 목록](results/requirements-lock.txt)을 참고합니다. macOS/Apple Silicon 실험 환경이며, 모든 플랫폼에서 동일하게 설치된다는 보장은 없습니다. `.venv`, Flutter 빌드/캐시, 로컬 실행 파일은 저장소에 포함하지 않습니다.

## 주요 파일

| 경로 | 내용 |
|---|---|
| `hanja_app/` | Flutter 앱 소스 및 앱용 모델 |
| `data/ai-free/` | AI-FREE 원본 분할 압축파일과 출처 기록 |
| `data/fonts-korean-hanja/` | 폰트 파일·라이선스·다운로드 기록 |
| `data/korean-hanja3500-v1/` | 3,500자 이미지 묶음, 정답·출처 목록, 학습/평가 구분 |
| `models/` | 비교용 공개 모델 |
| `results/hanja3500-head-only-v1/` | 출력층 학습 체크포인트, 특징 캐시, 평가 결과 |
| `scripts/` | 데이터 생성·학습·평가 스크립트 |
| `plans/` | 실험 계획서 |
| `vendor/` | 외부 프로젝트 서브모듈 |

큰 바이너리 및 5 MiB 이상 데이터/결과 파일은 `.gitattributes`에 따라 Git LFS로 관리합니다. 실패한 중간 생성본, 자동 생성 파일, 설치 로그는 제외합니다.

## 학습·평가

- [학습 계획서](plans/glyphweave-hanja3500-training-plan.md)
- [데이터 수량 및 학습/평가 분리](data/korean-hanja3500-v1/SPLIT.md)
- [글자별 데이터 수량](data/korean-hanja3500-v1/character-coverage.csv)
- [출력층만 학습한 결과](results/hanja3500-head-only-v1/REPORT.md)
- [Top-10 평가](results/hanja3500-head-only-v1/top10-evaluation.json)
- [최초 100자 비교](results/REPORT.md)

실제 평가 자료는 **SCUT-onHCCTestDB TradChar**이며 **SCUT-COUCH2009 Big5**와는 다릅니다. Noto Sans CJK KR 폰트는 평가 전용입니다. 글자별 데이터가 없으면 다른 글자로 대체하지 않고 없음으로 기록합니다.

## 출처와 이용 조건

원본 데이터·폰트·모델의 조건은 각각 유지되며, 저장소 전체에 하나의 새 라이선스를 부여하지 않습니다.

- AI-FREE: 원본 README에 CC BY-NC-SA 4.0 명시. [출처 기록](data/ai-free/README-pinned.md).
- SCUT: 원본 IME_Test README에 학술 연구용 공개 자료로 명시. [출처 기록](data/ime-readme.md).
- Noto/LXGW/은글꼴: 폰트별 첨부 라이선스 참조.
- 공개 모델 및 외부 코드는 각 원본 프로젝트의 이용 조건 참조.

AI-FREE 원본 필기자 정보와 전체 합성 여부는 독립적으로 검증하지 않았습니다. 평가 결과는 해당 분할의 측정치이며 모든 한국 한자 필기의 정확도를 보장하지 않습니다.
