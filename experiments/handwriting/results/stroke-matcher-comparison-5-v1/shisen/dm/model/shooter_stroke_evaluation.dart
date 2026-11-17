import 'handwriting.dart';

/// Snapshot of one pen-up decision, independent of the logging destination.
class ShooterStrokeEvaluation {
  ShooterStrokeEvaluation({
    this.positionRole = 'tie_break_only',
    this.assignments = const {},
    required this.reason,
    required this.character,
    required this.expectedStroke,
    required this.selectedStroke,
    required this.comboBefore,
    required this.comboAfter,
    required this.scoreDelta,
    required Iterable<InkPoint> input,
    required Iterable<ShooterStrokeCandidate> candidates,
  }) : input = List.unmodifiable(input),
       candidates = List.unmodifiable(candidates);

  final String reason;
  final String positionRole;
  final Map<int, int> assignments;
  final String? character;
  // Zero-based indices internally; the logger displays one-based stroke numbers.
  final int? expectedStroke, selectedStroke;
  final int comboBefore, comboAfter, scoreDelta;
  final List<InkPoint> input;
  final List<ShooterStrokeCandidate> candidates;
}

class ShooterStrokeCandidate {
  const ShooterStrokeCandidate({
    required this.enemyId,
    required this.character,
    required this.index,
    required this.alreadyAccepted,
    required this.forwardDistance,
    required this.reverseDistance,
    required this.positionDistance,
  });

  final int enemyId, index;
  final String character;
  final bool alreadyAccepted;
  final double forwardDistance, reverseDistance, positionDistance;
}
