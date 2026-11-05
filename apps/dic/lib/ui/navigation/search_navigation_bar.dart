import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

/// Matches Shadowing's floating menu dimensions and icon-caption layout.
class SearchNavigationBar extends StatefulWidget {
  const SearchNavigationBar({
    super.key,
    this.onRestart,
    this.onCamera,
    this.onSearch,
    this.cameraSelected = false,
  });
  final Future<void> Function()? onRestart;
  final VoidCallback? onCamera;
  final VoidCallback? onSearch;
  final bool cameraSelected;

  @override
  State<SearchNavigationBar> createState() => _SearchNavigationBarState();
}

class _SearchNavigationBarState extends State<SearchNavigationBar> {
  bool _restarting = false;

  Future<void> _restart() async {
    if (_restarting) return;
    setState(() => _restarting = true);
    try {
      await widget.onRestart!();
    } catch (_) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('재시작하지 못했습니다. 다시 시도해 주세요.')),
        );
      }
    } finally {
      if (mounted) setState(() => _restarting = false);
    }
  }

  Widget _label(IconData icon, String text) => Column(
    mainAxisSize: MainAxisSize.min,
    children: [
      Icon(icon, size: 26, color: const Color(0xffeeeeec)),
      const SizedBox(height: 4),
      Text(
        text,
        style: const TextStyle(
          fontSize: 10,
          height: 1,
          fontWeight: FontWeight.w600,
          color: Color(0xffeeeeec),
        ),
      ),
    ],
  );

  @override
  Widget build(BuildContext context) => SafeArea(
    top: false,
    maintainBottomViewPadding: true,
    minimum: const EdgeInsets.fromLTRB(20, 8, 20, 8),
    child: Center(
      heightFactor: 1,
      child: Container(
        constraints: const BoxConstraints(maxWidth: 440),
        height: 64,
        decoration: BoxDecoration(
          color: const Color(0xff151515),
          borderRadius: BorderRadius.circular(26),
          border: Border.all(color: const Color(0xff333333)),
        ),
        child: Row(
          children: [
            Expanded(
              child: Semantics(
                selected: !widget.cameraSelected,
                label: '검색',
                excludeSemantics: true,
                child: GestureDetector(
                  behavior: HitTestBehavior.opaque,
                  onTap: widget.onSearch,
                  child: Center(child: _label(Icons.search_rounded, '검색')),
                ),
              ),
            ),
            if (widget.onCamera != null)
              Expanded(
                child: InkWell(
                  onTap: widget.onCamera,
                  child: Center(
                    child: _label(Icons.camera_alt_outlined, '카메라'),
                  ),
                ),
              ),
            if (widget.onRestart != null)
              Expanded(
                child: Semantics(
                  button: true,
                  enabled: !_restarting,
                  label: '콜드 리로드',
                  excludeSemantics: true,
                  child: GestureDetector(
                    key: const ValueKey('development-restart'),
                    behavior: HitTestBehavior.opaque,
                    onTap: _restarting ? null : _restart,
                    child: Center(
                      child: _restarting
                          ? const CupertinoActivityIndicator()
                          : _label(Icons.refresh_rounded, '콜드 리로드'),
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    ),
  );
}
