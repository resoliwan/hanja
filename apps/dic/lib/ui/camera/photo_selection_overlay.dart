import 'package:flutter/material.dart';

class PhotoSelectionOverlay extends CustomPainter {
  const PhotoSelectionOverlay({
    required this.points,
    required this.frozen,
    this.frame,
  });
  final List<Offset> points;
  final bool frozen;
  final Rect? frame;
  static const brushWidth = 40.0;

  @override
  void paint(Canvas canvas, Size size) {
    final bounds = Offset.zero & size;
    if (!frozen) {
      final window = frame;
      final mask = Path()
        ..fillType = PathFillType.evenOdd
        ..addRect(bounds);
      if (window != null) {
        mask.addRRect(
          RRect.fromRectAndRadius(window, const Radius.circular(16)),
        );
      }
      canvas.drawPath(
        mask,
        Paint()..color = Colors.black.withValues(alpha: .45),
      );
      return;
    }
    canvas.saveLayer(bounds, Paint());
    canvas.drawRect(
      bounds,
      Paint()..color = Colors.black.withValues(alpha: .55),
    );
    if (points.isNotEmpty) {
      final reveal = Paint()
        ..blendMode = BlendMode.clear
        ..strokeWidth = brushWidth
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round
        ..style = PaintingStyle.stroke;
      final path = Path()..moveTo(points.first.dx, points.first.dy);
      for (final point in points.skip(1)) {
        path.lineTo(point.dx, point.dy);
      }
      if (points.length == 1) {
        canvas.drawCircle(
          points.first,
          brushWidth / 2,
          Paint()..blendMode = BlendMode.clear,
        );
      } else {
        canvas.drawPath(path, reveal);
      }
    }
    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant PhotoSelectionOverlay oldDelegate) => true;
}
