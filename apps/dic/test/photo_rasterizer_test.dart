import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:hanja_app/adapter/photo/photo_rasterizer.dart';
import 'package:hanja_app/dm/model/photo_selection.dart';

void main() {
  const selection = PhotoSelection(0, 0, 1, 1);
  test('Blank photos do not produce fabricated character candidates', () {
    final blank = img.Image(width: 100, height: 100);
    img.fill(blank, color: img.ColorRgb8(240, 240, 240));
    expect(
      () => normalizePhotoInk((
        bytes: Uint8List.fromList(img.encodePng(blank)),
        selection: selection,
      )),
      throwsStateError,
    );
  });

  test(
    'Photo ink uses the handwriting tensor scale across lighting levels',
    () {
      Float32List raster(int background, int ink) {
        final source = img.Image(width: 100, height: 100);
        img.fill(
          source,
          color: img.ColorRgb8(background, background, background),
        );
        img.fillRect(
          source,
          x1: 45,
          y1: 20,
          x2: 55,
          y2: 80,
          color: img.ColorRgb8(ink, ink, ink),
        );
        return normalizePhotoInk((
          bytes: Uint8List.fromList(img.encodePng(source)),
          selection: selection,
        ));
      }

      final bright = raster(255, 20), dim = raster(160, 40);
      expect(bright, hasLength(128 * 128));
      expect(bright.first, -1);
      expect(bright[64 * 128 + 64], 1);
      expect(dim[64 * 128 + 64], 1);
      final difference =
          List.generate(
            bright.length,
            (i) => (bright[i] - dim[i]).abs(),
          ).reduce((a, b) => a + b) /
          bright.length;
      expect(difference, lessThan(.02));
    },
  );
}
