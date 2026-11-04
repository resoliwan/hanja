import 'dart:convert';
import 'dart:typed_data';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

Uri dictionaryUri(String character) => Uri.parse(
  'https://hanja.dict.naver.com/#/search?range=all&query=${Uri.encodeComponent(character)}',
);

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

Future<Float32List> rasterize(List<List<ui.Offset>> strokes) async {
  final recorder = ui.PictureRecorder();
  final canvas = ui.Canvas(recorder);
  canvas.drawColor(const ui.Color(0xffffffff), ui.BlendMode.src);
  final paint = ui.Paint()
    ..color = const ui.Color(0xff000000)
    ..strokeWidth = 7
    ..strokeCap = ui.StrokeCap.round
    ..strokeJoin = ui.StrokeJoin.round
    ..style = ui.PaintingStyle.stroke;
  for (final stroke in strokes) {
    if (stroke.isEmpty) continue;
    if (stroke.length == 1) {
      canvas.drawCircle(
        stroke.first * 512,
        3.5,
        ui.Paint()..color = const ui.Color(0xff000000),
      );
      continue;
    }
    final path = ui.Path()
      ..moveTo(stroke.first.dx * 512, stroke.first.dy * 512);
    for (final p in stroke.skip(1)) {
      path.lineTo(p.dx * 512, p.dy * 512);
    }
    canvas.drawPath(path, paint);
  }
  final picture = recorder.endRecording();
  final image = await picture.toImage(512, 512);
  picture.dispose();
  final bytes = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
  image.dispose();
  return normalizeInk(bytes!.buffer.asUint8List(), 512, 512);
}

class HanjaRecognizer {
  late OrtSession session;
  late List<String> classes;
  Future<void> load() async {
    classes = List<String>.from(
      jsonDecode(await rootBundle.loadString('assets/classes.json')),
    );
    session = await OnnxRuntime().createSessionFromAsset(
      'assets/hanja3500.onnx',
      options: OrtSessionOptions(intraOpNumThreads: 2, interOpNumThreads: 1),
    );
  }

  Future<List<String>> predict(List<List<ui.Offset>> strokes) async {
    final input = await OrtValue.fromList(await rasterize(strokes), [
      1,
      1,
      128,
      128,
    ]);
    Map<String, OrtValue> outputs = {};
    try {
      outputs = await session.run({'image': input});
      final scores = await outputs['logits']!.asFlattenedList();
      final order = List.generate(classes.length, (i) => i)
        ..sort((a, b) => (scores[b] as num).compareTo(scores[a] as num));
      return order.take(10).map((i) => classes[i]).toList();
    } finally {
      await input.dispose();
      for (final v in outputs.values) {
        await v.dispose();
      }
    }
  }
}
