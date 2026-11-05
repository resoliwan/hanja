import 'dart:typed_data';

import 'package:image/image.dart' as img;

import '../../dm/model/photo_selection.dart';

Uint8List cropPhotoSquare(({Uint8List bytes, PhotoSelection selection}) input) {
  final decoded = img.decodeImage(input.bytes);
  if (decoded == null) throw const FormatException('Invalid photo');
  final source = img.bakeOrientation(decoded);
  final bounds = input.selection.square(source.width, source.height);
  final square = img.Image(width: bounds.side, height: bounds.side);
  img.fill(square, color: img.ColorRgb8(255, 255, 255));
  // Preserve a square even when the selected character touches an image edge.
  final left = bounds.x.clamp(0, source.width);
  final top = bounds.y.clamp(0, source.height);
  final right = (bounds.x + bounds.side).clamp(0, source.width);
  final bottom = (bounds.y + bounds.side).clamp(0, source.height);
  img.compositeImage(
    square,
    source,
    srcX: left,
    srcY: top,
    srcW: right - left,
    srcH: bottom - top,
    dstX: left - bounds.x,
    dstY: top - bounds.y,
    dstW: right - left,
    dstH: bottom - top,
  );
  final resized = bounds.side > 1600
      ? img.copyResize(square, width: 1600, height: 1600)
      : square;
  return Uint8List.fromList(img.encodeJpg(resized, quality: 95));
}
