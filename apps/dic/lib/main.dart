import 'adapter/dictionary/simpledic_dictionary.dart';

import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'config/dic_dependencies.dart';
import 'ui/hanja_app.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final readings = Map<String, String>.from(
    jsonDecode(await rootBundle.loadString('assets/hanja_readings.json'))
        as Map,
  );
  final definitions = await loadSimpleDicDictionary();
  final dependencies = DicDependencies();
  runApp(
    HanjaApp(
      readings: readings,
      definitions: definitions,
      bookmarks: dependencies.bookmarks,
      recognizePhoto: dependencies.recognizePhoto,
      onRestart: dependencies.restartDevelopmentApp,
      recognizeHandwriting: dependencies.recognizeHandwriting,
      openDictionary: dependencies.openDictionary,
    ),
  );
}
