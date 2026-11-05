import '../adapter/bookmarks/local_hanja_bookmarks.dart';
import '../out/hanja_bookmarks.dart';
import '../dm/usecase/recognize_photo.dart';

import 'package:flutter/foundation.dart';

import '../adapter/development/development_restart.dart';
import '../adapter/dictionary/naver_dictionary.dart';
import '../adapter/recognition/onnx_hanja_recognizer.dart';
import '../dm/usecase/open_hanja_dictionary.dart';
import '../dm/usecase/recognize_handwriting.dart';

class DicDependencies {
  DicDependencies() {
    final recognizer = OnnxHanjaRecognizer();
    recognizeHandwriting = RecognizeHandwriting(recognizer);
    recognizePhoto = RecognizePhoto(recognizer);
  }

  Future<void> Function()? get restartDevelopmentApp {
    const url = String.fromEnvironment('AOO_HOT_RESTART_URL');
    const token = String.fromEnvironment('AOO_HOT_RESTART_TOKEN');
    if (!kDebugMode || url.isEmpty || token.isEmpty) return null;
    return const DevelopmentRestart(url: url, token: token).call;
  }

  final HanjaBookmarks bookmarks = LocalHanjaBookmarks();

  late final RecognizePhoto recognizePhoto;

  late final RecognizeHandwriting recognizeHandwriting;
  final openDictionary = OpenHanjaDictionary(NaverDictionary());
}
