# GlyphWeave Hanja Flutter

Offline, single-character Korean hanja handwriting recognition for iOS, Android, and macOS. Tap any candidate to open that character's Naver Hanja Dictionary search in the default browser. Dictionary lookup requires an internet connection; recognition does not.

## Run

```sh
flutter pub get
flutter run -d macos
```

Targets: iOS 16+, macOS 14+, Android (Flutter project defaults). The model is included in assets; no Python server is needed.

## Model

`assets/hanja3500.onnx` is exported from `../results/hanja3500-head-only-v1/head-only-best.pt` (SHA256 `76be1904f5f3e7901a51cedc62f65bcecc1ab4e3bd765b562af3fba5311619a6`). It contains the frozen GlyphWeave feature extractor and the trained 3,500-class head. `assets/classes.json` preserves the checkpoint output order.

To regenerate with the existing project Python environment:

```sh
cd ..
.venv/bin/python scripts/export_hanja_mobile.py
```

Export requires PyTorch, ONNX and ONNX Runtime. The fixed input is float32 `[1,1,128,128]`, white background -1, black ink +1. Normalization crops pixels below 210, scales the longest side to 97 pixels, and centers the glyph on 128×128 white. The app draws only black ink on white for inference, excluding UI guides. Its separable bicubic resampler follows Pillow's kernel, with minor integer rounding differences covered by a fixture test. Existing dataset accuracy is not a guarantee for touchscreen handwriting.

## Validation

```sh
flutter analyze
flutter test test
flutter test integration_test/app_test.dart -d macos
# Substitute a simulator or emulator ID to check the mobile targets.
```

The integration test draws strokes through Flutter pointer input and runs the bundled native ONNX model, verifies 十, removes a stroke and verifies 一, then clears the canvas. The export script checks original/ONNX top-five agreement and score differences. Naver URL encoding and preprocessing parity have separate tests.

The application retains strokes in memory only, ignores stale recognition results after edits, and releases native inference tensors after each request. It shows ten candidates in two groups of five; model scores are not presented as calibrated probabilities.
