import 'package:flutter/material.dart';

import '../../out/hanja_bookmarks.dart';

class HanjaBookmarkSheet extends StatefulWidget {
  const HanjaBookmarkSheet({
    super.key,
    required this.store,
    required this.candidates,
    required this.readings,
    required this.onOpen,
  });
  final HanjaBookmarks store;
  final List<String> candidates;
  final Map<String, String> readings;
  final ValueChanged<String> onOpen;
  @override
  State<HanjaBookmarkSheet> createState() => _HanjaBookmarkSheetState();
}

class _HanjaBookmarkSheetState extends State<HanjaBookmarkSheet> {
  Set<String> _saved = {};
  bool _loading = true, _saving = false;
  String? _error;
  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final saved = await widget.store.read();
      if (mounted) {
        setState(() {
          _saved = saved;
          _loading = false;
        });
      }
    } catch (_) {
      if (mounted) {
        setState(() {
          _loading = false;
          _error = '북마크를 불러오지 못했습니다.';
        });
      }
    }
  }

  Future<void> _toggle(String character) async {
    if (_saving) return;
    setState(() => _saving = true);
    final updated = {..._saved};
    if (!updated.remove(character)) updated.add(character);
    try {
      await widget.store.write(updated);
      if (mounted) {
        setState(() {
          _saved = updated;
          _error = null;
        });
      }
    } catch (_) {
      if (mounted) setState(() => _error = '저장하지 못했습니다. 다시 시도해 주세요.');
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final characters = {...widget.candidates, ..._saved}.toList();
    return SafeArea(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Padding(
            padding: EdgeInsets.all(20),
            child: Text('북마크', style: TextStyle(fontSize: 20)),
          ),
          if (_loading)
            const Padding(
              padding: EdgeInsets.all(24),
              child: CircularProgressIndicator(),
            )
          else if (_error != null)
            Padding(padding: const EdgeInsets.all(16), child: Text(_error!)),
          if (!_loading && characters.isEmpty)
            const Padding(
              padding: EdgeInsets.all(24),
              child: Text('한자를 검색한 뒤 저장할 수 있어요'),
            ),
          Flexible(
            child: ListView.builder(
              shrinkWrap: true,
              itemCount: characters.length,
              itemBuilder: (context, index) {
                final character = characters[index];
                return ListTile(
                  leading: Text(
                    character,
                    style: const TextStyle(fontSize: 32),
                  ),
                  title: Text(widget.readings[character] ?? '훈음 정보 없음'),
                  onTap: () => widget.onOpen(character),
                  trailing: IconButton(
                    tooltip: _saved.contains(character) ? '북마크 해제' : '북마크 저장',
                    onPressed: _saving ? null : () => _toggle(character),
                    icon: Icon(
                      _saved.contains(character)
                          ? Icons.bookmark_rounded
                          : Icons.bookmark_border_rounded,
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
