import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

import '../../dm/model/handwriting.dart';
import '../../out/handwriting_recognizer.dart';
import '../../out/photo_recognizer.dart';
import '../../dm/model/photo_selection.dart';
import '../photo/photo_rasterizer.dart';
import 'ink_rasterizer.dart';

class OnnxHanjaRecognizer implements HandwritingRecognizer, PhotoRecognizer {
  OrtSession? _session;
  Future<void>? _loading;
  Future<List<String>>? _prediction;
  Future<void>? _disposing;
  bool _disposed = false;
  late List<String> _classes;
  @override
  Future<void> load() {
    if (_disposed) return Future.error(StateError('Recognizer is disposed'));
    return _loading ??= _load();
  }

  Future<void> _load() async {
    _classes = List<String>.from(
      jsonDecode(await rootBundle.loadString('assets/classes.json')),
    );
    _session = await OnnxRuntime().createSessionFromAsset(
      'assets/hanja.onnx',
      options: OrtSessionOptions(intraOpNumThreads: 2, interOpNumThreads: 1),
    );
  }

  @override
  Future<List<String>> predict(Handwriting handwriting) =>
      _predict(() => rasterize(handwriting));

  @override
  Future<List<String>> recognize(Uint8List image, PhotoSelection selection) =>
      _predict(() => rasterizePhoto(image, selection));

  Future<List<String>> _predict(Future<Float32List> Function() rasterizeInput) {
    if (_disposed) return Future.error(StateError('Recognizer is disposed'));
    if (_prediction != null) {
      return Future.error(StateError('Recognition is already running'));
    }
    final prediction = _run(rasterizeInput)
        .whenComplete(() => _prediction = null);
    _prediction = prediction;
    return prediction;
  }

  Future<List<String>> _run(
    Future<Float32List> Function() rasterizeInput,
  ) async {
    await load();
    final input = await OrtValue.fromList(await rasterizeInput(), [
      1,
      1,
      128,
      128,
    ]);
    Map<String, OrtValue> outputs = {};
    try {
      outputs = await _session!.run({'image': input});
      final scores = await outputs['logits']!.asFlattenedList();
      final order = List.generate(_classes.length, (i) => i)
        ..sort((a, b) => (scores[b] as num).compareTo(scores[a] as num));
      return order.take(10).map((i) => _classes[i]).toList();
    } finally {
      await input.dispose();
      for (final v in outputs.values) {
        await v.dispose();
      }
    }
  }

  @override
  Future<void> dispose() => _disposing ??= _dispose();

  Future<void> _dispose() async {
    _disposed = true;
    try {
      await _loading;
      await _prediction;
    } finally {
      final session = _session;
      _session = null;
      await session?.close();
    }
  }
}
