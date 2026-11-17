import 'dart:math' as math;

import '../model/handwriting.dart';

/// Per-stroke adaptation of HanziLookupJS's direction/length/center features.
/// Uses existing glyph paths, not character lookup or its bundled database.
abstract final class ShooterStrokeMatcher {
  static const acceptanceDistance = .28;

  /// Compare centers in the shared character square BEFORE normalizing shapes.
  /// Arc-length weighting avoids changes caused by pointer sampling speed.
  static double positionDistance(List<InkPoint> input, List<InkPoint> target) {
    final a = _center(input), b = _center(target);
    return a == null || b == null ? double.infinity : _distance(a, b);
  }

  static InkPoint? _center(List<InkPoint> points) {
    if (points.length < 2 ||
        points.any((p) => !p.x.isFinite || !p.y.isFinite)) {
      return null;
    }
    var length = 0.0, x = 0.0, y = 0.0;
    for (var i = 1; i < points.length; i++) {
      final a = points[i - 1], b = points[i];
      final weight = _distance(a, b);
      length += weight;
      x += (a.x + b.x) / 2 * weight;
      y += (a.y + b.y) / 2 * weight;
    }
    return length < .015 ? null : InkPoint(x / length, y / length);
  }

  /// Retry recognition preserves location and size as well as the drawn shape.
  static bool isSimilarRetry(List<InkPoint> input, List<InkPoint> previous) {
    if (positionDistance(input, previous) > .06) return false;
    double extent(List<InkPoint> points) {
      final xs = points.map((p) => p.x), ys = points.map((p) => p.y);
      return math.max(
        xs.reduce(math.max) - xs.reduce(math.min),
        ys.reduce(math.max) - ys.reduce(math.min),
      );
    }

    if (input.length < 2 || previous.length < 2) return false;
    final a = extent(input), b = extent(previous);
    if (math.min(a, b) / math.max(a, b) < .65) return false;
    return matches(input, previous);
  }

  static bool matches(List<InkPoint> input, List<InkPoint> target) =>
      distance(input, target) < acceptanceDistance;

  /// Lower is better. Translation and uniform scale do not affect the result;
  /// direction, turns and relative segment positions do.
  static double distance(List<InkPoint> input, List<InkPoint> target) {
    return _shapeDistance(input, target, requireEndpoints: true);
  }

  /// Writing allows slanted marks without the legacy endpoint hard veto.
  /// Direction and shape still contribute to the same acceptance score.
  static double writingDistance(List<InkPoint> input, List<InkPoint> target) =>
      _shapeDistance(input, target, requireEndpoints: false);

  static double _shapeDistance(
    List<InkPoint> input,
    List<InkPoint> target, {
    required bool requireEndpoints,
  }) {
    final a = _normalize(input), b = _normalize(target);
    if (a == null || b == null) return double.infinity;
    if (requireEndpoints &&
        (_distance(a.first, b.first) > .6 || _distance(a.last, b.last) > .6)) {
      return double.infinity;
    }
    final inputTurn = _majorTurn(a), targetTurn = _majorTurn(b);
    // Ignore small hooks, but keep a clear elbow distinct from a straight line.
    if ((inputTurn < .3 && targetTurn > 1.2) ||
        (targetTurn < .3 && inputTurn > 1.2)) {
      return double.infinity;
    }
    final left = _segments(a), right = _segments(b);
    // Sequence alignment tolerates a rounded corner or a small terminal hook.
    var previous = List<double>.generate(right.length + 1, (i) => i * .5);
    for (var i = 0; i < left.length; i++) {
      final current = List<double>.filled(right.length + 1, 0);
      current[0] = (i + 1) * .5;
      for (var j = 0; j < right.length; j++) {
        final x = left[i], y = right[j];
        final direction = (1 - math.cos(x.direction - y.direction)) / 2;
        final length =
            (x.length - y.length).abs() / math.max(x.length, y.length);
        final position = _distance(x.center, y.center);
        final cost = .65 * direction + .1 * length + .25 * position;
        current[j + 1] = math.min(
          previous[j] + cost,
          math.min(previous[j + 1], current[j]) + .5,
        );
      }
      previous = current;
    }
    return previous.last / math.max(left.length, right.length);
  }

  static double _majorTurn(List<InkPoint> points) {
    final first = math.atan2(
      points[8].y - points[0].y,
      points[8].x - points[0].x,
    );
    final last = math.atan2(
      points[23].y - points[15].y,
      points[23].x - points[15].x,
    );
    return math.atan2(math.sin(last - first), math.cos(last - first)).abs();
  }

  static List<_StrokeSegment> _segments(List<InkPoint> points) => [
    for (var i = 0; i < points.length - 1; i++)
      _StrokeSegment(points[i], points[i + 1]),
  ];

  static double _distance(InkPoint a, InkPoint b) =>
      math.sqrt(math.pow(a.x - b.x, 2) + math.pow(a.y - b.y, 2));

  static List<InkPoint>? _normalize(List<InkPoint> points) {
    if (points.length < 2 ||
        points.any((p) => !p.x.isFinite || !p.y.isFinite)) {
      return null;
    }
    final lengths = <double>[0];
    for (var i = 1; i < points.length; i++) {
      lengths.add(lengths.last + _distance(points[i - 1], points[i]));
    }
    if (lengths.last < .015) return null;
    final sampled = <InkPoint>[];
    var segment = 1;
    for (var i = 0; i < 24; i++) {
      final distance = lengths.last * i / 23;
      while (segment < points.length - 1 && lengths[segment] < distance) {
        segment++;
      }
      final span = lengths[segment] - lengths[segment - 1];
      final t = span == 0 ? 0.0 : (distance - lengths[segment - 1]) / span;
      sampled.add(
        InkPoint(
          points[segment - 1].x +
              (points[segment].x - points[segment - 1].x) * t,
          points[segment - 1].y +
              (points[segment].y - points[segment - 1].y) * t,
        ),
      );
    }
    final minX = sampled.map((p) => p.x).reduce(math.min);
    final minY = sampled.map((p) => p.y).reduce(math.min);
    final width = sampled.map((p) => p.x).reduce(math.max) - minX;
    final height = sampled.map((p) => p.y).reduce(math.max) - minY;
    final scale = math.max(width, height);
    if (scale < .01) return null;
    return sampled
        .map(
          (p) => InkPoint(
            (p.x - minX - width / 2) / scale,
            (p.y - minY - height / 2) / scale,
          ),
        )
        .toList();
  }
}

class _StrokeSegment {
  _StrokeSegment(InkPoint start, InkPoint end)
    : direction = math.atan2(end.y - start.y, end.x - start.x),
      length = math.max(1e-9, ShooterStrokeMatcher._distance(start, end)),
      center = InkPoint((start.x + end.x) / 2, (start.y + end.y) / 2);
  final double direction, length;
  final InkPoint center;
}
