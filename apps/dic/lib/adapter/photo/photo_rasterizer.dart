import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

import '../../dm/model/photo_selection.dart';
import '../recognition/ink_normalization.dart';
import 'photo_crop.dart';

/// Use the display codec so HEIC and orientation match the visible photo.
Future<Float32List> rasterizePhoto(
  Uint8List bytes,
  PhotoSelection selection,
) async {
  final codec = await ui.instantiateImageCodec(bytes);
  try {
    final frame = await codec.getNextFrame();
    try {
      final png = await frame.image.toByteData(format: ui.ImageByteFormat.png);
      if (png == null) throw const FormatException('Invalid photo');
      return await compute(normalizePhotoInk, (
        bytes: png.buffer.asUint8List(png.offsetInBytes, png.lengthInBytes),
        selection: selection,
      ));
    } finally {
      frame.image.dispose();
    }
  } finally {
    codec.dispose();
  }
}

/// Otsu threshold separates dark ink from photographed paper or screens.
/// Then apply the same scale and tensor range used for drawn strokes.
Float32List normalizePhotoInk(
  ({Uint8List bytes, PhotoSelection selection}) input,
) {
  final gray = img.grayscale(img.decodeJpg(cropPhotoSquare(input))!);
  final histogram = List<int>.filled(256, 0);
  for (final pixel in gray) {
    histogram[pixel.r.toInt()]++;
  }
  final total = gray.width * gray.height;
  final sum = List.generate(
    256,
    (i) => i * histogram[i],
  ).fold<int>(0, (a, b) => a + b);
  var count = 0, partialSum = 0, threshold = 0;
  var bestVariance = -1.0;
  for (var i = 0; i < 255; i++) {
    count += histogram[i];
    partialSum += i * histogram[i];
    if (count == 0 || count == total) continue;
    final difference =
        partialSum / count - (sum - partialSum) / (total - count);
    final variance = count * (total - count) * difference * difference;
    if (variance > bestVariance) {
      bestVariance = variance;
      threshold = i;
    }
  }
  if (bestVariance <= 0) throw StateError('사진에서 글자를 찾지 못했습니다.');
  final rgba = Uint8List(total * 4);
  var offset = 0;
  for (final pixel in gray) {
    final value = pixel.r <= threshold ? 0 : 255;
    rgba[offset++] = value;
    rgba[offset++] = value;
    rgba[offset++] = value;
    rgba[offset++] = 255;
  }
  return normalizeInk(rgba, gray.width, gray.height);
}
