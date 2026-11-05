import 'dart:typed_data';

import 'package:hanja_app/ui/dictionary/hanja_definition_sheet.dart';

import 'package:flutter/foundation.dart'
    show
        debugDefaultTargetPlatformOverride,
        TargetPlatform,
        FlutterError,
        debugPrint;
import 'package:hanja_app/dm/model/photo_selection.dart';
import 'package:hanja_app/dm/usecase/recognize_photo.dart';
import 'package:hanja_app/out/photo_recognizer.dart';
import 'package:hanja_app/ui/camera/camera_background.dart';

import 'dart:async';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart' show TextField, AppBar, Scaffold;
import 'package:flutter_test/flutter_test.dart';
import 'package:hanja_app/dm/model/handwriting.dart';
import 'package:hanja_app/dm/usecase/recognize_handwriting.dart';
import 'package:hanja_app/dm/usecase/open_hanja_dictionary.dart';
import 'package:hanja_app/out/handwriting_recognizer.dart';
import 'package:hanja_app/out/external_dictionary.dart';
import 'package:hanja_app/ui/hanja_app.dart';

class _Recognizer implements HandwritingRecognizer {
  final result = Completer<List<String>>();
  Handwriting? input;
  bool disposed = false;
  @override
  Future<void> load() async {}
  @override
  Future<List<String>> predict(Handwriting handwriting) {
    input = handwriting;
    return result.future;
  }

  @override
  Future<void> dispose() async {
    disposed = true;
  }
}

class _Dictionary implements ExternalDictionary {
  String? openedCharacter;
  @override
  Future<void> open(String character) async {
    openedCharacter = character;
  }
}

class _PhotoRecognizer implements PhotoRecognizer {
  @override
  Future<List<String>> recognize(
    Uint8List image,
    PhotoSelection selection,
  ) async => ['天'];
}

void main() {
  test('Handwriting snapshot survives edits to the original strokes', () {
    final strokes = [
      [const InkPoint(.1, .2)],
    ];
    final handwriting = Handwriting(strokes);
    strokes.first.clear();
    strokes.clear();
    expect(handwriting.strokes.single.single.x, .1);
    expect(() => handwriting.strokes.first.clear(), throwsUnsupportedError);
  });

  test('Empty handwriting does not invoke the recognition port', () async {
    final recognizer = _Recognizer();
    expect(await RecognizeHandwriting(recognizer)(Handwriting([[]])), isEmpty);
    expect(recognizer.input, isNull);
  });

  testWidgets(
    'Font options resize candidates and restart invokes the session action',
    (tester) async {
      final recognizer = _Recognizer();
      var restarts = 0;
      await tester.pumpWidget(
        HanjaApp(
          readings: const {'一': '한 일'},
          recognizeHandwriting: RecognizeHandwriting(recognizer),
          openDictionary: OpenHanjaDictionary(_Dictionary()),
          onRestart: () async {
            restarts++;
          },
        ),
      );
      await tester.pumpAndSettle();
      expect(find.text('여기에 쓰세요'), findsOneWidget);
      expect(find.text('한자를 쓰면 후보가 나타납니다'), findsNothing);
      final canvas = tester.getRect(
        find.byKey(const ValueKey('writingCanvas')),
      );
      await tester.dragFrom(canvas.center, const Offset(30, 0));
      await tester.pump(const Duration(milliseconds: 400));
      recognizer.result.complete(['一', '二']);
      await tester.pumpAndSettle();
      final before = tester.widget<Text>(find.text('一')).style!.fontSize!;
      expect(
        tester
            .widget<GridView>(find.byKey(const ValueKey('candidate-list')))
            .physics,
        isA<NeverScrollableScrollPhysics>(),
      );
      expect(
        tester.getTopLeft(find.text('二')).dy,
        greaterThan(tester.getTopLeft(find.text('一')).dy),
      );
      expect(
        tester.getBottomLeft(find.byTooltip('전체 지우기')).dy,
        greaterThan(tester.getBottomLeft(find.text('一')).dy),
      );
      await tester.tap(find.byTooltip('옵션'));
      await tester.pumpAndSettle();
      tester
          .widget<CupertinoSlider>(
            find.byKey(const ValueKey('font-size-slider')),
          )
          .onChanged!(1.6);
      await tester.pump();
      await tester.tap(find.byTooltip('닫기'));
      await tester.pumpAndSettle();
      expect(tester.widget<Text>(find.text('一')).style!.fontSize, before * 1.6);
      expect(
        tester
            .widget<GridView>(find.byKey(const ValueKey('candidate-list')))
            .physics,
        isA<ClampingScrollPhysics>(),
      );
      expect(
        tester.getTopLeft(find.text('二')).dy,
        tester.getTopLeft(find.text('一')).dy,
      );
      expect(tester.takeException(), isNull);
      await tester.tap(find.byKey(const ValueKey('development-restart')));
      await tester.pumpAndSettle();
      expect(restarts, 1);
    },
  );

  testWidgets(
    'Typing Korean displays local candidates without opening the dictionary',
    (tester) async {
      final dictionary = _Dictionary();
      await tester.pumpWidget(
        HanjaApp(
          readings: const {'天': '하늘 천', '木': '나무 목'},
          recognizeHandwriting: RecognizeHandwriting(_Recognizer()),
          openDictionary: OpenHanjaDictionary(dictionary),
        ),
      );
      await tester.pumpAndSettle();
      await tester.enterText(find.byType(TextField), '  하늘  ');
      await tester.pump();
      expect(find.text('天'), findsOneWidget);
      expect(find.text('하늘 천'), findsOneWidget);
      expect(dictionary.openedCharacter, isNull);
      await tester.enterText(find.byType(TextField), '목');
      await tester.pump();
      expect(find.text('木'), findsOneWidget);
      expect(find.text('天'), findsNothing);
      await tester.tap(find.text('木'));
      await tester.pump();
      expect(dictionary.openedCharacter, isNull);
      expect(find.byType(HanjaDefinitionSheet), findsOneWidget);
      await tester.pumpAndSettle();
      await tester.tap(find.byTooltip('닫기'));
      await tester.pumpAndSettle();
      await tester.enterText(find.byType(TextField), '없는검색어');
      await tester.pump();
      expect(find.text('검색 결과가 없습니다'), findsOneWidget);
      await tester.enterText(find.byType(TextField), '');
      await tester.pump();
      expect(find.text('木'), findsNothing);
      expect(find.text('검색 결과가 없습니다'), findsNothing);
    },
  );

  testWidgets(
    'Camera switches the background while keeping search and navigation',
    (tester) async {
      debugDefaultTargetPlatformOverride = TargetPlatform.macOS;
      addTearDown(() => debugDefaultTargetPlatformOverride = null);
      await tester.pumpWidget(
        HanjaApp(
          recognizeHandwriting: RecognizeHandwriting(_Recognizer()),
          openDictionary: OpenHanjaDictionary(_Dictionary()),
          recognizePhoto: RecognizePhoto(_PhotoRecognizer()),
        ),
      );
      await tester.pumpAndSettle();
      final beforeFrame = tester.getRect(
        find.byKey(const ValueKey('writingCanvas')),
      );
      await tester.tap(find.text('카메라'));
      await tester.pumpAndSettle();
      expect(find.byType(CameraBackground), findsOneWidget);
      expect(
        tester.getRect(find.byType(CameraBackground)),
        tester.getRect(find.byType(Scaffold)),
      );
      expect(find.byType(TextField), findsOneWidget);
      expect(find.byType(AppBar), findsNothing);
      expect(
        tester.getRect(find.byKey(const ValueKey('cameraWritingFrame'))),
        beforeFrame,
      );
      expect(find.text('여기에 쓰세요'), findsNothing);
      expect(find.byTooltip('옵션'), findsOneWidget);
      await tester.tap(find.text('검색'));
      await tester.pumpAndSettle();
      expect(find.byType(CameraBackground), findsNothing);
      expect(find.text('여기에 쓰세요'), findsOneWidget);
      debugDefaultTargetPlatformOverride = null;
    },
  );

  testWidgets(
    'Two-line readings fit above the home indicator with large text',
    (tester) async {
      tester.view.physicalSize = const Size(393, 852);
      tester.view.devicePixelRatio = 1;
      tester.view.padding = const FakeViewPadding(top: 59, bottom: 34);
      tester.platformDispatcher.textScaleFactorTestValue = 1.3;
      final originalError = FlutterError.onError;
      FlutterError.onError = (details) {
        debugPrint(details.toString());
        originalError!(details);
      };
      addTearDown(() => FlutterError.onError = originalError);
      addTearDown(tester.view.reset);
      addTearDown(tester.platformDispatcher.clearTextScaleFactorTestValue);
      final recognizer = _Recognizer();
      await tester.pumpWidget(
        HanjaApp(
          readings: const {'天': '하늘 천\n하늘의 뜻'},
          recognizeHandwriting: RecognizeHandwriting(recognizer),
          openDictionary: OpenHanjaDictionary(_Dictionary()),
        ),
      );
      await tester.pumpAndSettle();
      final canvas = tester.getRect(
        find.byKey(const ValueKey('writingCanvas')),
      );
      await tester.dragFrom(canvas.center, const Offset(20, 0));
      await tester.pump(const Duration(milliseconds: 400));
      recognizer.result.complete(['天']);
      await tester.pumpAndSettle();
      expect(find.text('하늘 천\n하늘의 뜻'), findsOneWidget);
      expect(tester.takeException(), isNull);
      await tester.tap(find.byTooltip('옵션'));
      await tester.pumpAndSettle();
      tester
          .widget<CupertinoSlider>(
            find.byKey(const ValueKey('font-size-slider')),
          )
          .onChanged!(1.6);
      await tester.pump();
      await tester.tap(find.byTooltip('닫기'));
      await tester.pumpAndSettle();
      expect(tester.takeException(), isNull);
    },
  );

  testWidgets('Text size leaves the writing frame and hint in place', (
    tester,
  ) async {
    await tester.pumpWidget(
      HanjaApp(
        recognizeHandwriting: RecognizeHandwriting(_Recognizer()),
        openDictionary: OpenHanjaDictionary(_Dictionary()),
      ),
    );
    await tester.pumpAndSettle();
    final canvas = find.byKey(const ValueKey('writingCanvas'));
    final hint = find.text('여기에 쓰세요');
    final frameBefore = tester.getRect(canvas);
    final hintBefore = tester.getRect(hint);
    await tester.tap(find.byTooltip('옵션'));
    await tester.pumpAndSettle();
    tester
        .widget<CupertinoSlider>(find.byKey(const ValueKey('font-size-slider')))
        .onChanged!(1.6);
    await tester.pump();
    await tester.tap(find.byTooltip('닫기'));
    await tester.pumpAndSettle();
    expect(tester.getRect(canvas), frameBefore);
    expect(tester.getRect(hint), hintBefore);
    expect(tester.takeException(), isNull);
  });

  testWidgets('Keyboard lifts candidates without resizing the writing frame', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(393, 852);
    tester.view.devicePixelRatio = 1;
    tester.view.viewPadding = const FakeViewPadding(top: 59, bottom: 34);
    tester.view.padding = const FakeViewPadding(top: 59, bottom: 34);
    addTearDown(tester.view.reset);
    await tester.pumpWidget(
      HanjaApp(
        readings: const {'天': '하늘 천'},
        recognizeHandwriting: RecognizeHandwriting(_Recognizer()),
        openDictionary: OpenHanjaDictionary(_Dictionary()),
      ),
    );
    await tester.pumpAndSettle();
    final canvas = find.byKey(const ValueKey('writingCanvas'));
    final frameBefore = tester.getRect(canvas);
    final hintBefore = tester.getRect(find.text('여기에 쓰세요'));
    await tester.enterText(find.byType(TextField), '하늘');
    tester.view.viewInsets = const FakeViewPadding(bottom: 300);
    tester.view.padding = const FakeViewPadding(top: 59);
    await tester.pumpAndSettle();
    expect(tester.getRect(canvas), frameBefore);
    expect(tester.getRect(find.text('여기에 쓰세요')), hintBefore);
    expect(
      tester.getRect(find.byKey(const ValueKey('candidate-list'))).bottom,
      552,
    );
    expect(tester.takeException(), isNull);
  });

  for (final clearBeforeResult in [false, true]) {
    testWidgets(
      clearBeforeResult
          ? 'Clearing ink discards an in-flight recognition result'
          : 'Drawing recognizes through the port and opens the selected dictionary entry',
      (tester) async {
        final recognizer = _Recognizer();
        final dictionary = _Dictionary();
        await tester.pumpWidget(
          HanjaApp(
            readings: const {'一': '한 일'},
            recognizeHandwriting: RecognizeHandwriting(recognizer),
            openDictionary: OpenHanjaDictionary(dictionary),
          ),
        );
        await tester.pumpAndSettle();
        final canvas = find.byKey(const ValueKey('writingCanvas'));
        final rect = tester.getRect(canvas);
        await tester.dragFrom(
          rect.center - Offset(rect.width * .25, 0),
          Offset(rect.width * .5, 0),
        );
        await tester.pump(const Duration(milliseconds: 400));
        expect(recognizer.input!.strokes, hasLength(1));
        if (clearBeforeResult) {
          await tester.tap(find.byTooltip('전체 지우기'));
          await tester.pump();
        }
        recognizer.result.complete(['一']);
        await tester.pumpAndSettle();
        if (clearBeforeResult) {
          expect(find.text('一'), findsNothing);
          expect(find.text('여기에 쓰세요'), findsOneWidget);
        } else {
          expect(find.text('한 일'), findsOneWidget);
          await tester.tap(find.text('一'));
          await tester.pump();
          expect(dictionary.openedCharacter, isNull);
          expect(find.byType(HanjaDefinitionSheet), findsOneWidget);
        }
        await tester.pumpWidget(const SizedBox());
        expect(recognizer.disposed, isTrue);
      },
    );
  }
}
