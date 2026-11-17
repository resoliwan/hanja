import 'handwriting.dart';

class ShooterStroke {
  ShooterStroke(Iterable<InkPoint> path, Iterable<Iterable<InkPoint>> outlines)
    : path = List.unmodifiable(path),
      outlines = List.unmodifiable(
        outlines.map((p) => List<InkPoint>.unmodifiable(p)),
      );
  final List<InkPoint> path;
  final List<List<InkPoint>> outlines;
}

class ShooterGlyph {
  ShooterGlyph(this.character, this.label, Iterable<ShooterStroke> strokes)
    : strokes = List.unmodifiable(strokes);
  final String character, label;
  final List<ShooterStroke> strokes;
}
