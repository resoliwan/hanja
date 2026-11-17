import 'dart:math' as math;

import '../model/handwriting.dart';
import '../model/shooter_glyph.dart';
import '../model/shooter_stroke_evaluation.dart';
import '../service/shooter_stroke_matcher.dart';

enum WritingStrokeVerdict { accepted, pending, incorrect, ignored }

class WritingStrokeDecision {
  WritingStrokeDecision(
    this.verdict,
    Map<int, int> assignments,
    this.candidates, {
    this.selectedStroke,
  }) : assignments = Map.unmodifiable(assignments);

  final WritingStrokeVerdict verdict;

  /// Input index -> reference stroke index. Recomputed after each pen-up.
  final Map<int, int> assignments;
  final List<ShooterStrokeCandidate> candidates;
  final int? selectedStroke;
}

/// Owns writing recognition policy, ambiguity and error classification.
/// One instance per question; rendering and scoring consume its decision.
class EvaluateWritingStroke {
  EvaluateWritingStroke(this.glyph, {required this.questionId});

  final ShooterGlyph glyph;
  final int questionId;
  final _costs = <List<double>>[];
  Map<int, int> _assignments = {};
  static const _uncertainDistance = .36;
  // Small sampling/curvature differences must not reorder equivalent strokes.
  static const _equivalentShapeDistance = .01;

  WritingStrokeDecision call(List<InkPoint> input) {
    WritingStrokeDecision result(
      WritingStrokeVerdict verdict,
      List<ShooterStrokeCandidate> candidates, {
      int? selected,
    }) => WritingStrokeDecision(
      verdict,
      _assignments,
      candidates,
      selectedStroke: selected,
    );
    if (!ShooterStrokeMatcher.matches(input, input)) {
      // Taps, non-finite coordinates and negligible movement aren't mistakes.
      return result(WritingStrokeVerdict.ignored, const []);
    }
    final reverse = input.reversed.toList();
    final accepted = _assignments.values.toSet();
    final candidates = <ShooterStrokeCandidate>[
      for (var i = 0; i < glyph.strokes.length; i++)
        ShooterStrokeCandidate(
          enemyId: questionId,
          character: glyph.character,
          index: i,
          alreadyAccepted: accepted.contains(i),
          forwardDistance: ShooterStrokeMatcher.writingDistance(
            input,
            glyph.strokes[i].path,
          ),
          reverseDistance: ShooterStrokeMatcher.writingDistance(
            reverse,
            glyph.strokes[i].path,
          ),
          positionDistance: ShooterStrokeMatcher.positionDistance(
            input,
            glyph.strokes[i].path,
          ),
        ),
    ];
    final costs = [
      for (final c in candidates)
        math.min(c.forwardDistance, c.reverseDistance),
    ];
    if (costs.every((cost) => cost >= _uncertainDistance)) {
      return result(WritingStrokeVerdict.incorrect, candidates);
    }
    final previousCount = _assignments.length;
    _costs.add(costs);
    _assignments = _assign();
    final selected = _assignments[_costs.length - 1];
    return result(
      _assignments.length > previousCount
          ? WritingStrokeVerdict.accepted
          : WritingStrokeVerdict.pending,
      candidates,
      selected: selected,
    );
  }

  /// Rectangular Hungarian assignment: reference rows, input + dummy columns.
  /// Missing strokes cost more than all real matches together, so maximize
  /// coverage first, then minimize total shape error. No location/order veto.
  Map<int, int> _assign() {
    final n = glyph.strokes.length, m = _costs.length + glyph.strokes.length;
    final missing = n + 1.0;
    final rankedCosts = _costs.map(_rankShapeCandidates).toList();
    double cost(int row, int column) {
      if (column >= _costs.length) return missing;
      final distance = _costs[column][row];
      if (distance >= ShooterStrokeMatcher.acceptanceDistance) {
        return missing * 2;
      }
      // Order only breaks equivalent-shape ties; it never excludes a candidate.
      // Bound the total tie cost independently of input/question size.
      final orderDistance = (row - column).abs() / m;
      final tieCost =
          (orderDistance * orderDistance + row / (n * m)) * 1e-7 / n;
      return rankedCosts[column][row] + tieCost;
    }

    final u = List<double>.filled(n + 1, 0);
    final v = List<double>.filled(m + 1, 0);
    final p = List<int>.filled(m + 1, 0);
    final way = List<int>.filled(m + 1, 0);
    for (var i = 1; i <= n; i++) {
      p[0] = i;
      var j0 = 0;
      final minv = List<double>.filled(m + 1, double.infinity);
      final used = List<bool>.filled(m + 1, false);
      do {
        used[j0] = true;
        final i0 = p[j0];
        var delta = double.infinity, j1 = 0;
        for (var j = 1; j <= m; j++) {
          if (used[j]) continue;
          final current = cost(i0 - 1, j - 1) - u[i0] - v[j];
          if (current < minv[j]) {
            minv[j] = current;
            way[j] = j0;
          }
          if (minv[j] < delta) {
            delta = minv[j];
            j1 = j;
          }
        }
        for (var j = 0; j <= m; j++) {
          if (used[j]) {
            u[p[j]] += delta;
            v[j] -= delta;
          } else {
            minv[j] -= delta;
          }
        }
        j0 = j1;
      } while (p[j0] != 0);
      do {
        final j1 = way[j0];
        p[j0] = p[j1];
        j0 = j1;
      } while (j0 != 0);
    }
    return {
      for (var j = 1; j <= _costs.length; j++)
        if (p[j] != 0 && cost(p[j] - 1, j - 1) < missing) j - 1: p[j] - 1,
    };
  }

  /// Group from each band's best score rather than rounding into fixed bins:
  /// near-identical candidates must not split at an arbitrary bin boundary.
  List<double> _rankShapeCandidates(List<double> distances) {
    final ranked = List<double>.of(distances);
    final indices = [
      for (var i = 0; i < distances.length; i++)
        if (distances[i] < ShooterStrokeMatcher.acceptanceDistance) i,
    ]..sort((a, b) => distances[a].compareTo(distances[b]));
    double? band;
    for (final index in indices) {
      if (band == null || distances[index] - band > _equivalentShapeDistance) {
        band = distances[index];
      }
      ranked[index] = band;
    }
    return ranked;
  }
}
