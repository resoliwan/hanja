import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hanja_app/adapter/dictionary/simpledic_dictionary.dart';
import 'package:hanja_app/ui/dictionary/hanja_definition_sheet.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('All recognition characters have definitions in source order', () async {
    final entries = await loadSimpleDicDictionary();
    final classes =
        jsonDecode(await rootBundle.loadString('assets/classes.json')) as List;
    final source = jsonDecode(
      File('assets/hanja_dictionary.json').readAsStringSync(),
    ) as Map;
    expect(entries.length, 4998);
    expect(entries.keys.toSet(), classes.toSet());
    for (final entry in entries.values) {
      expect(entry.meanings, isNotEmpty, reason: entry.character);
      expect(entry.meanings.map((m) => m.definition).toList(), [
        for (final m in source[entry.character]['meanings']) m['definition'],
      ]);
    }
    final tree = entries['木']!;
    expect(tree.mnemonic, source['木']['formation']['mnemonic']);
    expect(tree.historicalForms.length, 5);
    expect(
      tree.historicalForms.first.images.first.url,
      source['木']['historicalForms'][0]['images'][0]['sourceUrl'],
    );
    expect(tree.compounds.first.examples, [
      for (final e in source['木']['compounds'][0]['examples']) e['text'],
    ]);
    expect(
      tree.explanations.map((e) => e.text),
      contains(source['木']['modernComponents']['explanation']),
    );
    expect(() => entries.clear(), throwsUnsupportedError);
  });

  testWidgets(
    'Definition sheet renders numbered meanings and external action',
    (tester) async {
      final entries = await tester.runAsync(loadSimpleDicDictionary);
      final entry = entries!['木']!;
      var opened = false;
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: HanjaDefinitionSheet(
              character: entry.character,
              definition: entry,
              onOpenExternal: () => opened = true,
            ),
          ),
        ),
      );
      final first = entry.meanings.first;
      final meaning = find.text(
        '1. ${first.short} (${first.reading})\n${first.definition}',
      );
      await tester.scrollUntilVisible(meaning, 150);
      expect(meaning, findsOneWidget);
      final external = find.text('네이버 한자사전에서 보기');
      await tester.scrollUntilVisible(external, 200);
      await tester.tap(external);
      expect(opened, isTrue);
      expect(tester.takeException(), isNull);
    },
  );
}
