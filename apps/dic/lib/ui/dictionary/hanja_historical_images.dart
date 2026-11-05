import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../../dm/model/hanja_definition.dart';

Future<void> openDictionarySource(BuildContext context, String url) async {
  try {
    if (await launchUrl(Uri.parse(url))) return;
  } catch (_) {
    // Leave the sheet available when the source cannot be opened.
  }
  if (context.mounted) {
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(const SnackBar(content: Text('원문을 열지 못했습니다. 다시 시도해 주세요.')));
  }
}

class HanjaHistoricalImages extends StatelessWidget {
  const HanjaHistoricalImages({super.key, required this.forms});
  final List<HanjaHistoricalForm> forms;

  @override
  Widget build(BuildContext context) => Wrap(
    spacing: 12,
    runSpacing: 20,
    children: [
      for (final form in forms)
        for (final image in form.images)
          SizedBox(
            width: 140,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(
                  form.label,
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
                const SizedBox(height: 8),
                InkWell(
                  onTap: () => showDialog<void>(
                    context: context,
                    builder: (context) => Dialog(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Align(
                            alignment: Alignment.topRight,
                            child: IconButton(
                              tooltip: '닫기',
                              onPressed: () => Navigator.pop(context),
                              icon: const Icon(Icons.close),
                            ),
                          ),
                          Flexible(
                            child: InteractiveViewer(
                              child: _HistoricalImage(
                                image: image,
                                referer: form.sourceUrl,
                                size: 360,
                              ),
                            ),
                          ),
                          Padding(
                            padding: const EdgeInsets.all(16),
                            child: Text(image.caption),
                          ),
                        ],
                      ),
                    ),
                  ),
                  child: _HistoricalImage(
                    image: image,
                    referer: form.sourceUrl,
                    size: 140,
                  ),
                ),
                const SizedBox(height: 6),
                Text(image.caption, style: const TextStyle(fontSize: 12)),
                TextButton(
                  onPressed: () =>
                      openDictionarySource(context, form.sourceUrl),
                  child: const Text('출처 보기'),
                ),
              ],
            ),
          ),
    ],
  );
}

class _HistoricalImage extends StatefulWidget {
  const _HistoricalImage({
    required this.image,
    required this.referer,
    required this.size,
  });
  final HanjaHistoricalImage image;
  final String referer;
  final double size;
  @override
  State<_HistoricalImage> createState() => _HistoricalImageState();
}

class _HistoricalImageState extends State<_HistoricalImage> {
  int _attempt = 0;
  @override
  Widget build(BuildContext context) => Container(
    width: widget.size,
    height: widget.size,
    padding: const EdgeInsets.all(8),
    color: Colors.white,
    child: Image.network(
      widget.image.url,
      key: ValueKey(_attempt),
      headers: {'Referer': widget.referer},
      fit: BoxFit.contain,
      semanticLabel: widget.image.caption,
      loadingBuilder: (context, child, progress) => progress == null
          ? child
          : const Center(child: CircularProgressIndicator()),
      errorBuilder: (context, error, stack) => Center(
        child: TextButton(
          onPressed: () async {
            await NetworkImage(
              widget.image.url,
              headers: {'Referer': widget.referer},
            ).evict();
            if (mounted) setState(() => _attempt++);
          },
          child: const Text('이미지 다시 받기'),
        ),
      ),
    ),
  );
}
