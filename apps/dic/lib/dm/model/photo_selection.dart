import 'dart:math' as math;

/// Bounds of the rubbed area in normalized, upright image coordinates.
class PhotoSelection {
  const PhotoSelection(this.left, this.top, this.right, this.bottom);
  final double left, top, right, bottom;

  ({int x, int y, int side}) square(int width, int height) {
    final x1 = left.clamp(0, 1) * width;
    final y1 = top.clamp(0, 1) * height;
    final x2 = right.clamp(0, 1) * width;
    final y2 = bottom.clamp(0, 1) * height;
    final side = math.max(1, math.max(x2 - x1, y2 - y1).ceil());
    return (
      x: ((x1 + x2 - side) / 2).floor(),
      y: ((y1 + y2 - side) / 2).floor(),
      side: side,
    );
  }
}
