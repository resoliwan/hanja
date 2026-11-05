import '../model/handwriting.dart';
import '../../out/handwriting_recognizer.dart';

class RecognizeHandwriting {
  const RecognizeHandwriting(this._recognizer);
  final HandwritingRecognizer _recognizer;

  Future<void> prepare() => _recognizer.load();
  Future<List<String>> call(Handwriting handwriting) {
    if (handwriting.isEmpty) return Future.value(const []);
    return _recognizer.predict(handwriting);
  }

  Future<void> dispose() => _recognizer.dispose();
}
