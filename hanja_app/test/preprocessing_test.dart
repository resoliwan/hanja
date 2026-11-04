import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:hanja_app/recognizer.dart';

void main() {
  test('Canvas normalization matches official GlyphTransform', () {
    final rgba = File('test/fixtures/ink.rgba').readAsBytesSync();
    final bytes = File('test/fixtures/expected.f32').readAsBytesSync();
    final expected = Float32List.view(bytes.buffer);
    final actual = normalizeInk(rgba, 64, 64);
    double maxError = 0;
    for (int i = 0; i < actual.length; i++) {
      final e = (actual[i] - expected[i]).abs();
      if (e > maxError) maxError = e;
    }
    expect(
      maxError,
      lessThanOrEqualTo(2 / 127.5),
      reason: 'At most two grayscale levels of bicubic rounding difference',
    );
  });
  test('Empty canvas is rejected', () {
    expect(
      () => normalizeInk(
        Uint8List(64 * 64 * 4)..fillRange(0, 64 * 64 * 4, 255),
        64,
        64,
      ),
      throwsStateError,
    );
  });
}
