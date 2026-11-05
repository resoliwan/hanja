import '../dm/model/handwriting.dart';

abstract interface class HandwritingRecognizer {
  Future<void> load();
  Future<List<String>> predict(Handwriting handwriting);
  Future<void> dispose();
}
