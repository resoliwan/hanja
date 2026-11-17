/// A point in the normalized handwriting canvas (0–1 on each axis).
class InkPoint {
  const InkPoint(this.x, this.y);
  final double x;
  final double y;
}

/// An immutable snapshot, independent of later edits to the canvas.
class Handwriting {
  Handwriting(Iterable<Iterable<InkPoint>> strokes)
    : strokes = List.unmodifiable(
        strokes.map((stroke) => List<InkPoint>.unmodifiable(stroke)),
      );

  final List<List<InkPoint>> strokes;
  bool get isEmpty => strokes.every((stroke) => stroke.isEmpty);
}
