import 'dart:ui' as ui;
import 'dart:typed_data';

import '../../dm/model/handwriting.dart';
import 'ink_normalization.dart';

Future<Float32List> rasterize(Handwriting handwriting) async {
  final recorder = ui.PictureRecorder();
  final canvas = ui.Canvas(recorder);
  canvas.drawColor(const ui.Color(0xffffffff), ui.BlendMode.src);
  final paint = ui.Paint()
    ..color = const ui.Color(0xff000000)
    ..strokeWidth = 7
    ..strokeCap = ui.StrokeCap.round
    ..strokeJoin = ui.StrokeJoin.round
    ..style = ui.PaintingStyle.stroke;
  for (final stroke in handwriting.strokes) {
    if (stroke.isEmpty) continue;
    if (stroke.length == 1) {
      canvas.drawCircle(
        ui.Offset(stroke.first.x * 512, stroke.first.y * 512),
        3.5,
        ui.Paint()..color = const ui.Color(0xff000000),
      );
      continue;
    }
    final path = ui.Path()..moveTo(stroke.first.x * 512, stroke.first.y * 512);
    for (final p in stroke.skip(1)) {
      path.lineTo(p.x * 512, p.y * 512);
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
