import 'package:flutter/material.dart';

class InkPainter extends CustomPainter {
  final List<List<Offset>> strokes;
  InkPainter(this.strokes, {this.strokeScale = 1, this.showFrame = true});
  final double strokeScale;
  final bool showFrame;
  @override
  void paint(Canvas canvas, Size size) {
    final frame = Paint()
      ..color = const Color(0xffeeeeec)
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;
    // Draw one rounded corner, then reflect it into the other three corners.
    if (showFrame) {
      for (final flipX in [false, true]) {
        for (final flipY in [false, true]) {
          canvas.save();
          canvas.translate(flipX ? size.width : 0, flipY ? size.height : 0);
          canvas.scale(flipX ? -1 : 1, flipY ? -1 : 1);
          canvas.drawPath(
            Path()
              ..moveTo(2, 34)
              ..lineTo(2, 22)
              ..quadraticBezierTo(2, 2, 22, 2)
              ..lineTo(34, 2),
            frame,
          );
          canvas.restore();
        }
      }
    }
    final ink = Paint()
      ..color = const Color(0xffeeeeec)
      ..strokeWidth = size.width * 7 / 512 * strokeScale
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..style = PaintingStyle.stroke;
    for (final s in strokes) {
      if (s.isEmpty) continue;
      if (s.length == 1) {
        canvas.drawCircle(
          Offset(s.first.dx * size.width, s.first.dy * size.height),
          ink.strokeWidth / 2,
          Paint()..color = ink.color,
        );
        continue;
      }
      final path = Path()
        ..moveTo(s.first.dx * size.width, s.first.dy * size.height);
      for (final p in s.skip(1)) {
        path.lineTo(p.dx * size.width, p.dy * size.height);
      }
      canvas.drawPath(path, ink);
    }
  }

  @override
  bool shouldRepaint(covariant InkPainter oldDelegate) => true;
}
