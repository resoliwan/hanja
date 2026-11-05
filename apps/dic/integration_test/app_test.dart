import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:hanja_app/main.dart' as app;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  testWidgets('Draw ten, recognize on device, undo and clear', (tester) async {
    app.main();
    await tester.pumpAndSettle();
    for (var i = 0; i < 60 && find.text('인식 후보').evaluate().isEmpty; i++) {
      await tester.pump(const Duration(seconds: 1));
    }
    expect(find.text('인식 후보'), findsOneWidget);
    final rect = tester.getRect(find.byKey(const ValueKey('writingCanvas')));
    Future<void> stroke(Offset a, Offset b) async {
      final g = await tester.startGesture(a);
      for (var i = 1; i <= 20; i++) {
        await g.moveTo(Offset.lerp(a, b, i / 20)!);
        await tester.pump(const Duration(milliseconds: 10));
      }
      await g.up();
    }

    await stroke(
      Offset(rect.left + rect.width * .2, rect.top + rect.height * .45),
      Offset(rect.left + rect.width * .8, rect.top + rect.height * .45),
    );
    await stroke(
      Offset(rect.left + rect.width * .5, rect.top + rect.height * .2),
      Offset(rect.left + rect.width * .5, rect.top + rect.height * .8),
    );
    for (var i = 0; i < 30 && find.text('十').evaluate().isEmpty; i++) {
      await tester.pump(const Duration(seconds: 1));
    }
    expect(find.text('十'), findsOneWidget);
    await tester.tap(find.text('한 획 지우기'));
    for (var i = 0; i < 30 && find.text('一').evaluate().isEmpty; i++) {
      await tester.pump(const Duration(seconds: 1));
    }
    expect(find.text('一'), findsOneWidget);
    await tester.tap(find.text('전체 지우기'));
    await tester.pumpAndSettle();
    expect(find.text('여기에 한 글자를 써 주세요'), findsOneWidget);
    expect(find.text('十'), findsNothing);
  });
}
