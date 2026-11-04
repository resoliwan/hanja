import 'dart:async';

import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import 'recognizer.dart';

void main() => runApp(const HanjaApp());

class HanjaApp extends StatelessWidget {
  const HanjaApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
    debugShowCheckedModeBanner: false,
    title: '한자 필기',
    theme: ThemeData(
      brightness: Brightness.dark,
      colorScheme: ColorScheme.fromSeed(
        seedColor: const Color(0xff7eaeff),
        brightness: Brightness.dark,
      ),
      scaffoldBackgroundColor: const Color(0xff171c22),
      useMaterial3: true,
    ),
    home: const WritingPage(),
  );
}

class WritingPage extends StatefulWidget {
  const WritingPage({super.key});
  @override
  State<WritingPage> createState() => _WritingPageState();
}

class _WritingPageState extends State<WritingPage> {
  final recognizer = HanjaRecognizer();
  final strokes = <List<Offset>>[];
  List<String> candidates = [];
  bool ready = false, busy = false, more = false;
  int revision = 0;
  int? pointer;
  Timer? debounce;
  String status = '인식 모델 준비 중…';
  @override
  void initState() {
    super.initState();
    load();
  }

  Future<void> load() async {
    try {
      await recognizer.load();
      if (!mounted) {
        await recognizer.session.close();
        return;
      }
      setState(() {
        ready = true;
        status = '한자를 쓰면 후보가 나타납니다';
      });
    } catch (e) {
      if (mounted) setState(() => status = '모델을 불러오지 못했습니다. 앱을 다시 실행해 주세요.');
      debugPrint('$e');
    }
  }

  void changed() {
    revision++;
    debounce?.cancel();
    candidates = [];
    more = false;
  }

  void schedule() {
    debounce?.cancel();
    if (strokes.isNotEmpty) {
      debounce = Timer(const Duration(milliseconds: 350), recognize);
    }
  }

  Future<void> recognize() async {
    if (!ready || busy || strokes.isEmpty || pointer != null) return;
    final version = revision;
    final snapshot = strokes.map((s) => List<Offset>.from(s)).toList();
    setState(() {
      busy = true;
      status = '인식 중…';
    });
    final watch = Stopwatch()..start();
    try {
      final result = await recognizer.predict(snapshot);
      if (mounted && version == revision) {
        setState(() {
          candidates = result;
          status =
              '${strokes.length}획 · ${watch.elapsedMilliseconds}ms · 후보를 누르면 네이버 사전으로 이동';
        });
      }
    } catch (e) {
      if (mounted && version == revision) {
        setState(() => status = '인식하지 못했습니다. 다시 써 주세요.');
      }
      debugPrint('$e');
    } finally {
      if (mounted) {
        setState(() => busy = false);
        if (version != revision) schedule();
      }
    }
  }

  void erase(bool all) {
    setState(() {
      changed();
      pointer = null;
      if (all) {
        strokes.clear();
      } else if (strokes.isNotEmpty) {
        strokes.removeLast();
      }
      status = strokes.isEmpty ? '한자를 쓰면 후보가 나타납니다' : '다시 인식합니다';
    });
    schedule();
  }

  Future<void> openDictionary(String char) async {
    try {
      if (!await launchUrl(
        dictionaryUri(char),
        mode: LaunchMode.externalApplication,
      )) {
        throw StateError('launch failed');
      }
    } catch (_) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('사전을 열지 못했습니다. 인터넷 연결과 브라우저를 확인해 주세요.')),
        );
      }
    }
  }

  @override
  void dispose() {
    debounce?.cancel();
    if (ready) unawaited(recognizer.session.close());
    super.dispose();
  }

  Widget action(
    IconData icon,
    String text,
    VoidCallback? onTap, {
    bool primary = false,
  }) => Expanded(
    child: Padding(
      padding: const EdgeInsets.all(5),
      child: FilledButton.tonal(
        onPressed: onTap,
        style: FilledButton.styleFrom(
          backgroundColor: primary
              ? const Color(0xff234c7b)
              : const Color(0xff2c3540),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 5),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [Icon(icon), const SizedBox(height: 7), Text(text)],
        ),
      ),
    ),
  );
  @override
  Widget build(BuildContext context) => Scaffold(
    body: SafeArea(
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 760),
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const Row(
                  children: [
                    Icon(Icons.draw_outlined, color: Color(0xffa7c8ff)),
                    SizedBox(width: 10),
                    Text(
                      '한자 필기',
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    Spacer(),
                    Text('3,500자', style: TextStyle(color: Color(0xffa7b3c4))),
                  ],
                ),
                const SizedBox(height: 7),
                const Text(
                  '손으로 쓰고, 한자를 찾아보세요.',
                  style: TextStyle(color: Color(0xffa7b3c4)),
                ),
                const SizedBox(height: 18),
                Container(
                  height: 82,
                  decoration: BoxDecoration(
                    color: const Color(0xff252d37),
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: candidates.isEmpty
                      ? Center(
                          child: Text(
                            ready ? '인식 후보' : '모델 준비 중…',
                            style: const TextStyle(color: Color(0xff8e9bab)),
                          ),
                        )
                      : Row(
                          children: [
                            for (final c
                                in candidates.skip(more ? 5 : 0).take(5))
                              Expanded(
                                child: TextButton(
                                  onPressed: () => openDictionary(c),
                                  child: Text(
                                    c,
                                    style: const TextStyle(
                                      fontSize: 34,
                                      color: Colors.white,
                                    ),
                                  ),
                                ),
                              ),
                            IconButton(
                              tooltip: more ? '앞 후보' : '후보 더 보기',
                              onPressed: () => setState(() => more = !more),
                              icon: Icon(
                                more ? Icons.chevron_left : Icons.chevron_right,
                              ),
                            ),
                          ],
                        ),
                ),
                const SizedBox(height: 14),
                Expanded(
                  child: LayoutBuilder(
                    builder: (context, box) {
                      final side = box.biggest.shortestSide;
                      return Center(
                        child: SizedBox(
                          width: side,
                          height: side,
                          child: ClipRRect(
                            borderRadius: BorderRadius.circular(16),
                            child: Listener(
                              key: const ValueKey('writingCanvas'),
                              onPointerDown: (e) {
                                if (pointer != null) return;
                                setState(() {
                                  changed();
                                  pointer = e.pointer;
                                  strokes.add([
                                    Offset(
                                      (e.localPosition.dx / side).clamp(0, 1),
                                      (e.localPosition.dy / side).clamp(0, 1),
                                    ),
                                  ]);
                                });
                              },
                              onPointerMove: (e) {
                                if (pointer != e.pointer) return;
                                setState(() {
                                  strokes.last.add(
                                    Offset(
                                      (e.localPosition.dx / side).clamp(0, 1),
                                      (e.localPosition.dy / side).clamp(0, 1),
                                    ),
                                  );
                                });
                              },
                              onPointerUp: (e) {
                                if (pointer == e.pointer) {
                                  pointer = null;
                                  schedule();
                                }
                              },
                              onPointerCancel: (e) {
                                if (pointer == e.pointer) {
                                  pointer = null;
                                  schedule();
                                }
                              },
                              child: CustomPaint(
                                painter: InkPainter(strokes),
                                child: strokes.isEmpty
                                    ? const Center(
                                        child: IgnorePointer(
                                          child: Text(
                                            '여기에 한 글자를 써 주세요',
                                            style: TextStyle(
                                              color: Color(0xff8995a3),
                                              fontSize: 16,
                                            ),
                                          ),
                                        ),
                                      )
                                    : const SizedBox.expand(),
                              ),
                            ),
                          ),
                        ),
                      );
                    },
                  ),
                ),
                const SizedBox(height: 10),
                SizedBox(
                  height: 36,
                  child: Center(
                    child: Text(
                      status,
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        fontSize: 12,
                        color: Color(0xffa7b3c4),
                      ),
                    ),
                  ),
                ),
                Row(
                  children: [
                    action(
                      Icons.undo,
                      '한 획 지우기',
                      strokes.isEmpty ? null : () => erase(false),
                    ),
                    action(
                      Icons.delete_outline,
                      '전체 지우기',
                      strokes.isEmpty ? null : () => erase(true),
                    ),
                    action(
                      Icons.search,
                      '인식',
                      ready && !busy && strokes.isNotEmpty ? recognize : null,
                      primary: true,
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    ),
  );
}

class InkPainter extends CustomPainter {
  final List<List<Offset>> strokes;
  InkPainter(this.strokes);
  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawColor(const Color(0xff333d48), BlendMode.src);
    final guide = Paint()
      ..color = const Color(0xff5c6672)
      ..strokeWidth = 1;
    for (double p = 0; p < size.width; p += 10) {
      canvas.drawLine(
        Offset(p, size.height / 2),
        Offset(p + 3, size.height / 2),
        guide,
      );
      canvas.drawLine(
        Offset(size.width / 2, p),
        Offset(size.width / 2, p + 3),
        guide,
      );
    }
    final ink = Paint()
      ..color = const Color(0xfff5f7fa)
      ..strokeWidth = size.width * 7 / 512
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..style = PaintingStyle.stroke;
    for (final s in strokes) {
      if (s.isEmpty) continue;
      if (s.length == 1) {
        canvas.drawCircle(
          Offset(s.first.dx * size.width, s.first.dy * size.height),
          ink.strokeWidth / 2,
          Paint()..color = ink.color,
        );
        continue;
      }
      final path = Path()
        ..moveTo(s.first.dx * size.width, s.first.dy * size.height);
      for (final p in s.skip(1)) {
        path.lineTo(p.dx * size.width, p.dy * size.height);
      }
      canvas.drawPath(path, ink);
    }
  }

  @override
  bool shouldRepaint(covariant InkPainter oldDelegate) => true;
}
