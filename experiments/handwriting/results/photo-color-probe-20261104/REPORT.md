# Photo color recognition probe

Two glyphs (超, 譯) from IMG_4431.HEIC; synthetic variants, not a population accuracy benchmark.

Fixed soft glyph mask: clip((source luminance - 120) / 100, 0, 1). Uniform backgrounds: light=245, dark=55. Primaries: RGB 255/0. RGB gradient: red→green→blue horizontally. Center glare: 80% white blending at center, Gaussian width 0.18 × crop width, applied to foreground and background. This reconstruction simplifies original texture; original crops are separate controls.

Model: apps/dic/assets/hanja.onnx, SHA256 999c5f74c3339ffd7e1de140f37be104ee2a6171f8478ef127ed68f8744b9655

All tensors generated using Flutter image package and actual app normalizeInk/normalizePhotoInk functions. Direct-gray skips bounding-box normalization; gray-normalized uses existing handwriting normalization. Auto-polarity uses Otsu + majority of crop perimeter to decide which intensity class is background; it is experimental and can fail with gradients or mixed contrast.

|Case|Direct gray|Gray + existing normalization|Current binary|Auto polarity + binary|
|---|---|---|---|---|
|light-neutral|1/1|1/1|1/1|1/1|
|light-red|1/1|1/1|1/1|1/1|
|light-blue|1/1|1/1|1/1|1/1|
|light-green|1/1|1/1|1/1|1/1|
|light-rgb-gradient|1/1|1/1|1/1|1/1|
|light-red-center-glare|1/1|1/1|2/1|2/1|
|light-blue-center-glare|1/1|1/1|2/1|2/1|
|light-green-center-glare|1/1|1/1|2/1|2/1|
|dark-neutral|192/109|1846/3443|501/4414|1/1|
|dark-red|4594/3326|832/3587|483/4410|1/1|
|dark-blue|4478/2837|1405/4442|1/1|1/1|
|dark-green|4050/37|729/4198|502/4419|1/1|
|dark-rgb-gradient|4329/1646|1865/3785|619/3898|1/1|
|dark-red-center-glare|416/399|1117/3014|648/1242|1505/2146|
|dark-blue-center-glare|383/709|885/3423|514/1355|1442/1724|
|dark-green-center-glare|1724/884|244/3335|768/1318|1068/389|
|original|238/200|1886/3364|466/4388|1/1|

Each cell gives expected-class ranks in order 超 / 譯.

## Synthetic variant totals

{
  "direct-gray": {
    "top1": 16,
    "top10": 16,
    "total": 32
  },
  "gray-normalized": {
    "top1": 16,
    "top10": 16,
    "total": 32
  },
  "current-binary": {
    "top1": 15,
    "top10": 18,
    "total": 32
  },
  "auto-polarity-binary": {
    "top1": 23,
    "top10": 26,
    "total": 32
  }
}

Reproduce: generate_fixtures.py; copy export_tensors_test.dart to apps/dic/test/photo_color_probe_tmp_test.dart, run flutter test from apps/dic, then evaluate.py. Remove temporary test after run. Scripts use absolute workspace paths.