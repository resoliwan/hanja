# 맥 한자 필기 인식기 4종 실행 결과

**실행 완료: 네 인식기 모두 실제 학습 모델로 동일한 100개 필기를 인식했습니다.**

**데이터 변경:** SCUT-COUCH2009 Big5 원본은 공식 서버 접속 시간 초과로 확보하지 못했습니다. 아래 결과는 별도 공개 데이터셋인 **SCUT-onHCCTestDB TradChar**입니다. COUCH2009 결과로 해석하면 안 됩니다.

| 인식기 | Top-1 | Top-5 | 오류 | 중앙 처리시간/자 |
|---|---:|---:|---:|---:|
| Hanzi Lookup | 35/100 | 49/100 | 1 | 1.2 ms |
| Zinnia | 23/100 | 29/100 | 0 | 15.6 ms |
| cnn_chinese_hw (legacy) | 38/100 | 51/100 | 0 | 6.3 ms |
| GlyphWeave | 95/100 | 100/100 | 0 | 87.5 ms |

## 데이터와 평가 방법

- 출처: https://github.com/HCIILAB/IME_Test (연구용 공개 자료).
- `onHCCTestDB-TradChar 1.pot`의 파일 순서상 첫 100개. 서로 다른 정답 100자이며 필터링·추가 학습·간체/번체 변환을 하지 않았습니다.
- 별도 UTF-8 정답 파일과 POT 내부 문자 코드(GBK)를 100개 모두 대조했습니다. Big5는 글자 집합을 가리키며 POT 코드 바이트 자체는 GBK입니다.
- 획 구분, 원래 획순, 획 내부 좌표 순서를 보존했습니다. Hanzi Lookup의 u8 좌표 제한에 맞춰 종횡비를 보존해 0~255 안으로 축소했습니다. Zinnia는 같은 정규화 좌표를 사용했습니다.
- cnn_chinese_hw는 원래 제공된 28×28×3 directMap 전처리를 사용했습니다. GlyphWeave는 같은 좌표를 512×512 흰 바탕/검은 선(선 두께 12px)으로 그린 뒤 128×128로 축소하고 공식 전처리를 적용했습니다.
- Top-1은 첫 후보 일치, Top-5는 첫 다섯 후보 안에 정답 포함. 오류도 분모 100에 포함합니다.
- 시간은 단일 실행의 중앙값이며 모델 로딩을 제외하고 각 어댑터의 전처리/인식을 포함합니다. 엄밀한 성능 측정용 반복 실험은 아닙니다. GlyphWeave는 CPU 4스레드, CNN은 TensorFlow Lite CPU, Hanzi Lookup은 Node WASM, Zinnia는 arm64 네이티브입니다.

## 버전과 주의점

- `cnn_chinese_hw` 최신판은 PyTorch로 변경되었고 사전학습 체크포인트가 없습니다. 실제 비교에는 공개 TFLite 모델이 남아 있는 커밋 `9049eb5e7cb353b19d07241cb8712e77c18afad3`를 사용했습니다. 최신판 성능을 뜻하지 않습니다.
- GlyphWeave v1.0.0 공식 inference 패키지의 SHA-256이 공식 SHA256SUMS.txt와 일치함을 확인했습니다.
- Zinnia는 소스로 맥 arm64 라이브러리를 빌드하고 Tegaki Traditional Chinese 0.3 모델을 사용했습니다.
- Hanzi Lookup은 `gugray/hanzi_lookup`에 동봉된 WASM 모델을 사용했습니다.
- 97번 `髯`에는 한 점짜리 획이 있어 Hanzi Lookup의 공개 WASM이 인덱스 오류로 중단되었습니다. 해당 샘플은 실패 처리하고 모델 상태를 다시 만든 뒤 다음 샘플을 처리했습니다.
- Zinnia·CNN·GlyphWeave의 문자 목록에는 정답 100자가 모두 포함됩니다. Hanzi Lookup의 전체 문자 목록 대조는 수행하지 않았습니다.
- 첫 100자에 대한 소규모 점검입니다. 전체 번체/한국 한자 성능이나 필기자 일반화 성능으로 확대해석할 수 없으며 모델 학습 자료와의 중복 여부를 별도로 검증하지 않았습니다.
- SCUT-DLVCLab/DOLPHIN의 COUCH09.zip도 다운로드했으나 이는 정답 문자 라벨이 없는 필기자 검색용 자료여서 이번 정확도 계산에는 사용하지 않았습니다.

## 재실행

작업 폴더에서:
```sh
.venv/bin/python scripts/extract_samples.py
.venv/bin/python scripts/benchmark.py data/tradchar-first100.json --out results/tradchar-first100
.venv/bin/python scripts/make_report.py
```

100개별 그림과 후보: `results/report.html` / 원시 결과: `results/tradchar-first100/predictions.csv` / 출처·표본: `data/sample-manifest.json` / 환경: `results/environment.json`.
