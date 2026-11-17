// Staged beside a byte-identical snapshot of the Shisen domain sources.
import 'dart:convert';
import 'dart:io';
import 'shisen/dm/model/handwriting.dart';
import 'shisen/dm/model/shooter_glyph.dart';
import 'shisen/dm/usecase/evaluate_writing_stroke.dart';

List<InkPoint> points(dynamic data) => [
  for(final p in data) InkPoint((p[0] as num).toDouble(),(p[1] as num).toDouble())
];

void main(List<String> args) {
  final output=File(args[1]).openWrite();
  for(final line in File(args[0]).readAsLinesSync()) {
    final row=jsonDecode(line);
    final glyph=ShooterGlyph(row['character'],row['character'],[
      for(final s in row['template']) ShooterStroke(points(s),[])
    ]);
    final evaluator=EvaluateWritingStroke(glyph,questionId:0);
    final storedInputs=<int>[];
    final prefix=<dynamic>[];
    final watch=Stopwatch()..start();
    for(var i=0;i<row['strokes'].length;i++) {
      final decision=evaluator(points(row['strokes'][i]));
      if(decision.verdict==WritingStrokeVerdict.accepted || decision.verdict==WritingStrokeVerdict.pending) {
        storedInputs.add(i+1);
      }
      prefix.add({
        'verdict':decision.verdict.name,
        'relations':[
          for(final e in decision.assignments.entries) [storedInputs[e.key],e.value+1]
        ]
      });
    }
    watch.stop();
    output.writeln(jsonEncode({'case_id':row['case_id'],'prefix':prefix,
      'relations':prefix.last['relations'],'runtime_seconds':watch.elapsedMicroseconds/1e6}));
  }
  output.close();
}
