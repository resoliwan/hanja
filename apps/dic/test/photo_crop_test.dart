import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:hanja_app/adapter/photo/photo_crop.dart';
import 'package:hanja_app/dm/model/photo_selection.dart';

void main() {
  test(
    'Portrait selection expands around the selected center into a square',
    () {
      final square = const PhotoSelection(.25, .25, .75, .75).square(100, 200);
      expect(square, (x: 0, y: 50, side: 100));
    },
  );

  test(
    'Crop preserves selected pixels rather than shrinking the whole photo',
    () {
      final source = img.Image(width: 100, height: 200);
      img.fill(source, color: img.ColorRgb8(255, 0, 0));
      img.fillRect(
        source,
        x1: 30,
        y1: 70,
        x2: 69,
        y2: 109,
        color: img.ColorRgb8(0, 0, 255),
      );
      final crop = img.decodeJpg(
        cropPhotoSquare((
          bytes: Uint8List.fromList(img.encodePng(source)),
          selection: const PhotoSelection(.3, .35, .7, .55),
        )),
      )!;
      expect(crop.width, crop.height);
      final center = crop.getPixel(crop.width ~/ 2, crop.height ~/ 2);
      expect(center.b, greaterThan(240));
      expect(center.r, lessThan(10));
    },
  );

  test('Selection touching an edge uses white padding without stretching', () {
    final source = img.Image(width: 100, height: 200);
    img.fill(source, color: img.ColorRgb8(0, 0, 0));
    final crop = img.decodeJpg(
      cropPhotoSquare((
        bytes: Uint8List.fromList(img.encodePng(source)),
        selection: const PhotoSelection(0, .25, .2, .75),
      )),
    )!;
    expect(crop.width, 100);
    expect(crop.height, 100);
    expect(crop.getPixel(5, 50).r, greaterThan(240));
    expect(crop.getPixel(60, 50).r, lessThan(10));
  });
}
