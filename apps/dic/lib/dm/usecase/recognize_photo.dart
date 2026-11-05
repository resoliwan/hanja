import 'dart:typed_data';

import '../model/photo_selection.dart';
import '../../out/photo_recognizer.dart';

class RecognizePhoto {
  const RecognizePhoto(this._recognizer);
  final PhotoRecognizer _recognizer;
  Future<List<String>> call(Uint8List image, PhotoSelection selection) =>
      _recognizer.recognize(image, selection);
}
