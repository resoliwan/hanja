import 'package:flutter/material.dart';

import 'writing_options_sheet.dart';

class WritingOptionsButton extends StatelessWidget {
  const WritingOptionsButton({
    super.key,
    required this.scale,
    required this.onChanged,
    required this.strokeScale,
    required this.onStrokeChanged,
  });
  final double scale;
  final double strokeScale;
  final ValueChanged<double> onStrokeChanged;
  final ValueChanged<double> onChanged;

  @override
  Widget build(BuildContext context) => IconButton(
    tooltip: '옵션',
    icon: const Icon(Icons.more_vert),
    onPressed: () => showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      showDragHandle: true,
      backgroundColor: const Color(0xff1c1c1e),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
      ),
      clipBehavior: Clip.antiAlias,
      builder: (_) => WritingOptionsSheet(
        scale: scale,
        strokeScale: strokeScale,
        onChanged: onChanged,
        onStrokeChanged: onStrokeChanged,
      ),
    ),
  );
}
