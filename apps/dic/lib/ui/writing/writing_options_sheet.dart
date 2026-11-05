import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

import 'ink_painter.dart';

/// Inset, grouped settings in the same bottom-sheet layout as Shadowing.
class WritingOptionsSheet extends StatefulWidget {
  const WritingOptionsSheet({
    super.key,
    required this.scale,
    required this.strokeScale,
    required this.onChanged,
    required this.onStrokeChanged,
  });
  final double scale, strokeScale;
  final ValueChanged<double> onChanged, onStrokeChanged;

  @override
  State<WritingOptionsSheet> createState() => _WritingOptionsSheetState();
}

class _WritingOptionsSheetState extends State<WritingOptionsSheet> {
  late double _scale = widget.scale;
  late double _strokeScale = widget.strokeScale;

  @override
  Widget build(BuildContext context) => SafeArea(
    top: false,
    child: SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(20, 0, 20, 24),
      child: CupertinoTheme(
        data: const CupertinoThemeData(brightness: Brightness.dark),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              children: [
                const Expanded(
                  child: Text(
                    '보기 설정',
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.w700,
                      color: Color(0xffeeeeec),
                    ),
                  ),
                ),
                IconButton(
                  tooltip: '닫기',
                  onPressed: () => Navigator.pop(context),
                  style: IconButton.styleFrom(
                    backgroundColor: const Color(0xff2c2c2e),
                    foregroundColor: const Color(0xffaeaeb2),
                    fixedSize: const Size(32, 32),
                  ),
                  icon: const Icon(Icons.close_rounded, size: 20),
                ),
              ],
            ),
            const SizedBox(height: 16),
            _SettingGroup(
              title: '폰트 크기',
              value: '${(_scale * 100).round()}%',
              icon: CupertinoIcons.textformat_size,
              child: Column(
                children: [
                  SizedBox(
                    height:
                        22 +
                        MediaQuery.textScalerOf(context).scale(36 * 1.6) * 1.2 +
                        MediaQuery.textScalerOf(context).scale(11 * 1.6) * 1.3,
                    child: Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text(
                            '天',
                            key: const ValueKey('font-preview-character'),
                            style: TextStyle(
                              fontSize: 36 * _scale,
                              height: 1.2,
                              fontWeight: FontWeight.w300,
                              color: const Color(0xffeeeeec),
                            ),
                          ),
                          const SizedBox(height: 6),
                          Text(
                            '하늘 천',
                            key: const ValueKey('font-preview-reading'),
                            style: TextStyle(
                              fontSize: 11 * _scale,
                              height: 1.3,
                              color: const Color(0xffb5b5b5),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  Row(
                    children: [
                      const Text(
                        '가',
                        style: TextStyle(
                          fontSize: 14,
                          color: Color(0xffaeaeb2),
                        ),
                      ),
                      Expanded(
                        child: CupertinoSlider(
                          key: const ValueKey('font-size-slider'),
                          value: _scale,
                          min: .8,
                          max: 1.6,
                          divisions: 8,
                          onChanged: (value) {
                            setState(() => _scale = value);
                            widget.onChanged(value);
                          },
                        ),
                      ),
                      const Text(
                        '가',
                        style: TextStyle(
                          fontSize: 24,
                          color: Color(0xffaeaeb2),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            _SettingGroup(
              title: '획 굵기',
              value: '${(_strokeScale * 100).round()}%',
              icon: CupertinoIcons.pencil,
              child: Column(
                children: [
                  Row(
                    children: [
                      const Text(
                        '가늘게',
                        style: TextStyle(
                          fontSize: 13,
                          color: Color(0xffaeaeb2),
                        ),
                      ),
                      Expanded(
                        child: CupertinoSlider(
                          key: const ValueKey('stroke-width-slider'),
                          value: _strokeScale,
                          min: .5,
                          max: 3,
                          divisions: 10,
                          onChanged: (value) {
                            setState(() => _strokeScale = value);
                            widget.onStrokeChanged(value);
                          },
                        ),
                      ),
                      const Text(
                        '굵게',
                        style: TextStyle(
                          fontSize: 13,
                          color: Color(0xffaeaeb2),
                        ),
                      ),
                    ],
                  ),
                  SizedBox(
                    height: 88,
                    width: double.infinity,
                    child: CustomPaint(
                      key: const ValueKey('stroke-preview'),
                      painter: InkPainter(
                        const [
                          [Offset(.34, .15), Offset(.65, .15)],
                          [Offset(.3, .4), Offset(.69, .4)],
                          [
                            Offset(.5, .15),
                            Offset(.49, .4),
                            Offset(.45, .65),
                            Offset(.34, .88),
                          ],
                          [Offset(.49, .4), Offset(.54, .63), Offset(.66, .88)],
                        ],
                        strokeScale: _strokeScale,
                        showFrame: false,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    ),
  );
}

class _SettingGroup extends StatelessWidget {
  const _SettingGroup({
    required this.title,
    required this.value,
    required this.icon,
    required this.child,
  });
  final String title, value;
  final IconData icon;
  final Widget child;

  @override
  Widget build(BuildContext context) => DecoratedBox(
    decoration: BoxDecoration(
      color: const Color(0xff2c2c2e),
      borderRadius: BorderRadius.circular(18),
    ),
    child: Column(
      children: [
        Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              Icon(icon, size: 22, color: const Color(0xffeeeeec)),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  title,
                  style: const TextStyle(
                    fontSize: 17,
                    color: Color(0xffeeeeec),
                  ),
                ),
              ),
              Text(
                value,
                style: const TextStyle(fontSize: 17, color: Color(0xffaeaeb2)),
              ),
            ],
          ),
        ),
        const Divider(
          height: 1,
          thickness: .5,
          indent: 16,
          color: Color(0xff48484a),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
          child: child,
        ),
      ],
    ),
  );
}
