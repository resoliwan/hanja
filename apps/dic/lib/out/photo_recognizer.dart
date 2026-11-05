import 'dart:typed_data';

import '../dm/model/photo_selection.dart';

abstract interface class PhotoRecognizer {
  Future<List<String>> recognize(Uint8List image, PhotoSelection selection);
}
