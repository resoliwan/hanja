import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:hanja_app/config/dic_dependencies.dart';
import 'package:hanja_app/dm/model/handwriting.dart';
import 'package:hanja_app/dm/model/photo_selection.dart';

// Supply PHOTO_TEST_PATH to test a local photo without committing private data.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  const path = String.fromEnvironment('PHOTO_TEST_PATH');
  testWidgets('Recognizes the selected handwritten 天 in a real photo', (
    tester,
  ) async {
    final dependencies = DicDependencies();
    try {
      final candidates = await dependencies.recognizePhoto(
        await File(path).readAsBytes(),
        const PhotoSelection(.32, .28, .57, .52),
      );
      // ignore: avoid_print
      print('Photo candidates: $candidates');
      expect(candidates.first, '天');
      final handwriting = await dependencies.recognizeHandwriting(
        Handwriting([
          [const InkPoint(.2, .5), const InkPoint(.8, .5)],
          [const InkPoint(.5, .2), const InkPoint(.5, .8)],
        ]),
      );
      expect(handwriting.first, '十');
    } finally {
      await dependencies.recognizeHandwriting.dispose();
    }
  }, skip: path.isEmpty);
}
