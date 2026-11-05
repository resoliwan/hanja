import '../dm/model/hanja_definition.dart';
import '../out/hanja_bookmarks.dart';
import '../dm/usecase/recognize_photo.dart';

import 'package:flutter/material.dart';

import '../dm/usecase/recognize_handwriting.dart';
import '../dm/usecase/open_hanja_dictionary.dart';
import 'writing/writing_page.dart';

class HanjaApp extends StatelessWidget {
  const HanjaApp({
    super.key,
    this.readings = const {},
    this.definitions = const {},
    this.bookmarks,
    this.onRestart,
    this.recognizePhoto,
    required this.recognizeHandwriting,
    required this.openDictionary,
  });
  final HanjaBookmarks? bookmarks;
  final RecognizePhoto? recognizePhoto;
  final Map<String, String> readings;
  final Map<String, HanjaDefinition> definitions;
  final Future<void> Function()? onRestart;
  final RecognizeHandwriting recognizeHandwriting;
  final OpenHanjaDictionary openDictionary;
  @override
  Widget build(BuildContext context) => MaterialApp(
    debugShowCheckedModeBanner: false,
    title: '한자 필기',
    theme: ThemeData(
      brightness: Brightness.dark,
      colorScheme: ColorScheme.fromSeed(
        seedColor: const Color(0xff00754a),
        primary: const Color(0xff00754a),
        onPrimary: Colors.white,
        brightness: Brightness.dark,
      ),
      scaffoldBackgroundColor: const Color(0xff191919),
      useMaterial3: true,
    ),
    home: WritingPage(
      readings: readings,
      definitions: definitions,
      bookmarks: bookmarks,
      recognizePhoto: recognizePhoto,
      onRestart: onRestart,
      recognizeHandwriting: recognizeHandwriting,
      openDictionary: openDictionary,
    ),
  );
}
