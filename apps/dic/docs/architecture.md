# 앱 아키텍처

사천성과 같은 `dm / out / adapter / ui / config` 계층을 사용합니다.

- `dm/model`: Flutter 타입을 사용하지 않는 필기 좌표와 변경 불가능한 필기 스냅샷.
- `dm/usecase`: 필기 인식과 외부 한자 사전 열기. 구현체 대신 `out` 인터페이스를 호출합니다.
- `out`: 필기 인식 및 외부 사전 포트. Flutter·ONNX·URL 플러그인에 의존하지 않습니다.
- `adapter/recognition`: 필기 이미지 렌더링, 모델 입력 정규화, ONNX 추론과 자원 해제.
- `adapter/dictionary`: 네이버 사전 URL 생성 및 외부 브라우저 실행.
- `ui`: 앱 테마, 필기 화면, 화면용 캔버스 렌더링. 포인터·지연 인식·화면 상태는 화면에서 관리합니다.
- `config`: 실제 어댑터를 만들고 유스케이스에 생성자로 주입합니다.
- `main.dart`: 구성된 의존성으로 앱을 시작합니다.

호출 흐름: `ui → dm/usecase → out ← adapter`.
`dm`과 `out`은 순수 Dart이며, `ui`에서 어댑터나 플러그인을 직접 호출하지 않습니다.
화면의 Offset은 유스케이스 호출 전에 InkPoint로 변환됩니다. 인식 요청에는 깊은 복사된
스냅샷을 사용하고, 요청 이후 획이 변경되면 이전 인식 결과를 화면에 반영하지 않습니다.

필기 화면은 주입된 인식기의 수명을 소유합니다. 화면 종료 시 dispose를 호출하며,
ONNX 어댑터는 모델 로딩과 진행 중인 추론을 기다린 뒤 세션을 닫습니다.
새 화면에는 새 인식기를 주입합니다. 네이티브 세션과 텐서는 어댑터 밖에 노출하지 않습니다.

`flutter analyze`와 `flutter test test`로 분석·정규화·포트 주입·오래된 결과 무시를 확인합니다.
실제 모델 검증은 `flutter test integration_test/app_test.dart -d macos`로 실행합니다.
