import 'dart:math' as math;
import 'dart:typed_data';

// Pillow-compatible bicubic kernel, including antialiasing on downsampling.
double _cubic(double x) {
  x = x.abs();
  if (x < 1) return ((1.5 * x - 2.5) * x) * x + 1;
  if (x < 2) return ((-0.5 * x + 2.5) * x - 4) * x + 2;
  return 0;
}

List<int> _resize(List<int> src, int w, int h, int nw, int nh) {
  List<int> pass(
    List<int> input,
    int width,
    int height,
    int target,
    bool horizontal,
  ) {
    final old = horizontal ? width : height;
    final scale = old / target, filter = math.max(1.0, old / target);
    final result = List<int>.filled(
      horizontal ? target * height : width * target,
      255,
    );
    for (var t = 0; t < target; t++) {
      final center = (t + 0.5) * scale;
      final lo = math.max(0, (center - 2 * filter + 0.5).floor());
      final hi = math.min(old, (center + 2 * filter + 0.5).floor());
      final weights = [
        for (var i = lo; i < hi; i++) _cubic((i - center + 0.5) / filter),
      ];
      final sum = weights.fold(0.0, (a, b) => a + b);
      for (var other = 0; other < (horizontal ? height : width); other++) {
        double value = 0;
        for (var i = lo; i < hi; i++) {
          value +=
              input[horizontal ? other * width + i : i * width + other] *
              weights[i - lo] /
              sum;
        }
        result[horizontal ? other * target + t : t * width + other] = value
            .round()
            .clamp(0, 255);
      }
    }
    return result;
  }

  return pass(pass(src, w, h, nw, true), nw, h, nh, false);
}

Float32List normalizeInk(Uint8List rgba, int w, int h) {
  int left = w, top = h, right = -1, bottom = -1;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      if (rgba[(y * w + x) * 4] < 210) {
        left = math.min(left, x);
        right = math.max(right, x);
        top = math.min(top, y);
        bottom = math.max(bottom, y);
      }
    }
  }
  if (right < left) throw StateError('한자를 써 주세요.');
  final cw = right - left + 1,
      ch = bottom - top + 1,
      ratio = 97 / math.max(right - left + 1, bottom - top + 1);
  final nw = math.max(1, (cw * ratio).round()),
      nh = math.max(1, (ch * ratio).round());
  final crop = [
    for (var y = top; y <= bottom; y++)
      for (var x = left; x <= right; x++) rgba[(y * w + x) * 4],
  ];
  final resized = _resize(crop, cw, ch, nw, nh);
  final tensor = Float32List(128 * 128)..fillRange(0, 128 * 128, -1);
  final dx = (128 - nw) ~/ 2, dy = (128 - nh) ~/ 2;
  for (var y = 0; y < nh; y++) {
    for (var x = 0; x < nw; x++) {
      tensor[(y + dy) * 128 + x + dx] = 1 - resized[y * nw + x] / 127.5;
    }
  }
  return tensor;
}
