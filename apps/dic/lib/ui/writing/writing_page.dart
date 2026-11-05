import '../dictionary/hanja_definition_sheet.dart';
import '../../dm/model/hanja_definition.dart';
import '../bookmarks/hanja_bookmark_sheet.dart';
import '../../out/hanja_bookmarks.dart';
import '../camera/camera_background.dart';
import '../../dm/usecase/recognize_photo.dart';

import 'dart:async';

import 'package:flutter/material.dart';

import '../../dm/model/handwriting.dart';
import '../../dm/usecase/recognize_handwriting.dart';
import '../../dm/usecase/search_hanja_readings.dart';
import '../../dm/usecase/open_hanja_dictionary.dart';
import 'ink_painter.dart';
import 'dictionary_search_field.dart';
import 'writing_options_button.dart';
import '../navigation/search_navigation_bar.dart';

class WritingPage extends StatefulWidget {
  const WritingPage({
    super.key,
    this.readings = const {},
    this.definitions = const {},
    this.bookmarks,
    this.onRestart,
    this.recognizePhoto,
    required this.recognizeHandwriting,
    required this.openDictionary,
  });
  final HanjaBookmarks? bookmarks;
  final RecognizePhoto? recognizePhoto;
  final Map<String, String> readings;
  final Map<String, HanjaDefinition> definitions;
  final Future<void> Function()? onRestart;
  final RecognizeHandwriting recognizeHandwriting;
  final OpenHanjaDictionary openDictionary;
  @override
  State<WritingPage> createState() => _WritingPageState();
}

class _WritingPageState extends State<WritingPage> {
  final _writingFrameKey = GlobalKey();
  final _cameraController = CameraBackgroundController();
  bool _cameraMode = false;

  void _cameraUpdated() {
    if (!mounted || !_cameraMode) return;
    setState(() {
      candidates = _cameraController.candidates;
      status = _cameraController.message;
    });
  }

  void _setCameraMode(bool enabled) {
    FocusScope.of(context).unfocus();
    setState(() {
      changed();
      _cameraMode = enabled;
      _searchController.clear();
      status = '';
    });
    if (!enabled) schedule();
  }

  final _searchController = TextEditingController();
  bool get _searching => _searchController.text.trim().isNotEmpty;
  final strokes = <List<Offset>>[];
  List<String> candidates = [];
  bool ready = false, busy = false;
  double _fontScale = 1;
  double _strokeScale = 1;
  int get _candidateRows => _scrollCandidates ? 1 : 2;
  bool get _scrollCandidates => _fontScale > 1;
  double _candidateRowHeight(BuildContext context, {double? scale}) {
    final scaler = MediaQuery.textScalerOf(context);
    final textScale = scale ?? _fontScale;
    // Each rendered line can round up independently (notably iOS CJK fonts).
    // Reserve both reading lines before adding padding and the inter-text gap.
    return 22 +
        (scaler.scale(36 * textScale) * 1.2).ceilToDouble() +
        (scaler.scale(11 * textScale) * 1.3).ceilToDouble() * 2;
  }

  int revision = 0;
  int? pointer;
  Timer? debounce;
  String status = '인식 모델 준비 중…';
  @override
  void initState() {
    super.initState();
    _cameraController.addListener(_cameraUpdated);
    load();
  }

  Future<void> load() async {
    try {
      await widget.recognizeHandwriting.prepare();
      if (!mounted) {
        return;
      }
      setState(() {
        ready = true;
        status = '';
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
  }

  void _search(String query) {
    setState(() {
      _cameraMode = false;
      changed();
      candidates = SearchHanjaReadings(widget.readings)(query);
      status = _searching && candidates.isEmpty ? '검색 결과가 없습니다' : '';
    });
    if (!_searching) schedule();
  }

  void schedule() {
    debounce?.cancel();
    if (!_cameraMode && !_searching && strokes.isNotEmpty) {
      debounce = Timer(const Duration(milliseconds: 350), recognize);
    }
  }

  Future<void> recognize() async {
    if (_cameraMode ||
        _searching ||
        !ready ||
        busy ||
        strokes.isEmpty ||
        pointer != null) {
      return;
    }
    final version = revision;
    final snapshot = Handwriting(
      strokes.map(
        (stroke) => stroke.map((point) => InkPoint(point.dx, point.dy)),
      ),
    );
    setState(() {
      busy = true;
      status = '인식 중…';
    });

    try {
      final result = await widget.recognizeHandwriting(snapshot);
      if (mounted && version == revision) {
        setState(() {
          candidates = result;
          status = result.isEmpty ? '인식하지 못했습니다. 다시 써 주세요.' : '';
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
      _searchController.clear();
      pointer = null;
      if (all) {
        strokes.clear();
      } else if (strokes.isNotEmpty) {
        strokes.removeLast();
      }
      status = strokes.isEmpty ? '' : '다시 인식합니다';
    });
    schedule();
  }

  Future<void> openDictionary(String char) async {
    FocusScope.of(context).unfocus();
    await showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      showDragHandle: true,
      backgroundColor: const Color(0xff252525),
      builder: (_) => HanjaDefinitionSheet(
        character: char,
        definition: widget.definitions[char],
        onOpenExternal: () => _openExternalDictionary(char),
      ),
    );
  }

  Future<void> _openExternalDictionary(String char) async {
    try {
      await widget.openDictionary(char);
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
    _searchController.dispose();
    _cameraController.removeListener(_cameraUpdated);
    _cameraController.dispose();
    unawaited(
      widget.recognizeHandwriting.dispose().catchError((Object error) {
        debugPrint('$error');
      }),
    );
    super.dispose();
  }

  void _openBookmarks() {
    final store = widget.bookmarks;
    if (store == null) return;
    showModalBottomSheet<void>(
      context: context,
      backgroundColor: const Color(0xff252525),
      builder: (_) => HanjaBookmarkSheet(
        store: store,
        candidates: List.of(candidates),
        readings: widget.readings,
        onOpen: openDictionary,
      ),
    );
  }

  Widget _actionIcon(
    IconData icon,
    String label,
    VoidCallback? onPressed, {
    bool primary = false,
  }) => Expanded(
    child: SizedBox(
      height: 88,
      child: Center(
        child: IconButton(
          tooltip: label,
          onPressed: onPressed,
          style: IconButton.styleFrom(
            fixedSize: Size.square(primary ? 80 : 56),
            padding: EdgeInsets.zero,
            shape: const CircleBorder(),
            backgroundColor: primary
                ? Theme.of(context).colorScheme.onPrimary
                : const Color(0xff292929),
            disabledBackgroundColor: primary
                ? Theme.of(context).colorScheme.onPrimary
                : const Color(0xff292929),
            foregroundColor: primary
                ? Theme.of(context).colorScheme.primary
                : const Color(0xffeeeeec),
            disabledForegroundColor: const Color(0xff777777),
          ),
          icon: Icon(icon, size: primary ? 54 : 28),
        ),
      ),
    ),
  );

  Widget _actionRow() => Row(
    key: const ValueKey('candidate-actions'),
    children: [
      _actionIcon(
        Icons.bookmark_border_rounded,
        '북마크',
        widget.bookmarks == null ? null : _openBookmarks,
      ),
      if (_cameraMode) ...[
        _actionIcon(
          Icons.camera_alt_outlined,
          _cameraController.hasPhoto ? '다시 촬영' : '촬영',
          _cameraController.hasPhoto
              ? _cameraController.retake
              : (_cameraController.canCapture
                    ? _cameraController.capture
                    : null),
          primary: true,
        ),
        _actionIcon(
          Icons.photo_library_outlined,
          '사진 열기',
          _cameraController.recognizing ? null : _cameraController.pickPhoto,
        ),
      ] else ...[
        _actionIcon(
          Icons.close_rounded,
          '전체 지우기',
          strokes.isEmpty ? null : () => erase(true),
          primary: true,
        ),
        _actionIcon(
          Icons.undo_rounded,
          '한 획 지우기',
          strokes.isEmpty ? null : () => erase(false),
        ),
      ],
    ],
  );

  Widget _candidateList(BuildContext context) => SizedBox(
    height: _candidateRowHeight(context, scale: 1) * 2,
    child: candidates.isEmpty
        ? Center(
            child: Text(
              (_cameraMode
                      ? _cameraController.recognizing
                      : busy && !_searching)
                  ? '인식 중…'
                  : status,
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 12, color: Color(0xff999999)),
            ),
          )
        : LayoutBuilder(
            builder: (context, box) => GridView.builder(
              key: const ValueKey('candidate-list'),
              padding: EdgeInsets.zero,
              scrollDirection: Axis.horizontal,
              physics: _scrollCandidates
                  ? const ClampingScrollPhysics()
                  : const NeverScrollableScrollPhysics(),
              itemCount: _scrollCandidates || candidates.length <= 10
                  ? candidates.length
                  : 10,
              gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: _candidateRows,
                mainAxisExtent:
                    (box.maxWidth / 5) * (_scrollCandidates ? _fontScale : 1),
              ),
              itemBuilder: (context, index) {
                final character = candidates[index];
                final reading = widget.readings[character] ?? '훈음 정보 없음';
                return Align(
                  alignment: Alignment.topCenter,
                  child: SizedBox(
                    width: double.infinity,
                    child: TextButton(
                      onPressed: () => openDictionary(character),
                      style: TextButton.styleFrom(
                        shape: const RoundedRectangleBorder(),
                        padding: const EdgeInsets.symmetric(
                          horizontal: 3,
                          vertical: 8,
                        ),
                        foregroundColor: const Color(0xffeeeeec),
                      ),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text(
                            character,
                            style: TextStyle(
                              fontSize: 36 * _fontScale,
                              height: 1.2,
                              fontWeight: FontWeight.w300,
                            ),
                          ),
                          const SizedBox(height: 6),
                          Tooltip(
                            message: reading,
                            child: Text(
                              reading,
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                fontSize: 11 * _fontScale,
                                color: Color(0xffb5b5b5),
                                height: 1.3,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
  );

  @override
  Widget build(BuildContext context) => Scaffold(
    extendBody: _cameraMode,
    resizeToAvoidBottomInset: false,
    bottomNavigationBar: SearchNavigationBar(
      onRestart: widget.onRestart,
      cameraSelected: _cameraMode,
      onSearch: () => _setCameraMode(false),
      onCamera: widget.recognizePhoto == null
          ? null
          : () => _setCameraMode(true),
    ),
    body: LayoutBuilder(
      builder: (context, bodyConstraints) => Stack(
        children: [
          if (_cameraMode)
            Positioned.fill(
              child: CameraBackground(
                controller: _cameraController,
                recognizePhoto: widget.recognizePhoto!,
              ),
            ),
          SafeArea(
            // Scaffold supplies the floating menu inset when the camera extends
            // behind it, keeping controls and the recognition frame in place.
            bottom: _cameraMode,
            child: Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 600),
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(24, 24, 24, 8),
                  child: Column(
                    children: [
                      Row(
                        children: [
                          Expanded(
                            child: DictionarySearchField(
                              controller: _searchController,
                              onChanged: _search,
                            ),
                          ),
                          const SizedBox(width: 8),
                          WritingOptionsButton(
                            scale: _fontScale,
                            strokeScale: _strokeScale,
                            onStrokeChanged: (value) =>
                                setState(() => _strokeScale = value),
                            onChanged: (value) =>
                                setState(() => _fontScale = value),
                          ),
                        ],
                      ),
                      const SizedBox(height: 12),
                      Expanded(
                        child: LayoutBuilder(
                          builder: (context, box) {
                            // Keep the recognition coordinate system square on every screen.
                            final side = box.maxWidth < box.maxHeight
                                ? box.maxWidth
                                : box.maxHeight;
                            if (_cameraMode) {
                              WidgetsBinding.instance.addPostFrameCallback((_) {
                                final frame =
                                    _writingFrameKey.currentContext
                                            ?.findRenderObject()
                                        as RenderBox?;
                                if (mounted && _cameraMode && frame != null) {
                                  _cameraController.setPreviewFrame(
                                    frame.localToGlobal(Offset.zero) &
                                        frame.size,
                                  );
                                }
                              });
                            }
                            return Center(
                              child: SizedBox(
                                key: _writingFrameKey,
                                width: side,
                                height: side,
                                child: Stack(
                                  children: [
                                    Positioned.fill(
                                      child: _cameraMode
                                          ? IgnorePointer(
                                              child: CustomPaint(
                                                key: const ValueKey(
                                                  'cameraWritingFrame',
                                                ),
                                                painter: InkPainter(const []),
                                              ),
                                            )
                                          : Listener(
                                              key: const ValueKey(
                                                'writingCanvas',
                                              ),
                                              behavior: HitTestBehavior.opaque,
                                              onPointerDown: (e) {
                                                if (pointer != null) return;
                                                setState(() {
                                                  changed();
                                                  _searchController.clear();
                                                  FocusScope.of(context)
                                                      .unfocus();
                                                  pointer = e.pointer;
                                                  strokes.add([
                                                    Offset(
                                                      (e.localPosition.dx /
                                                              side)
                                                          .clamp(0, 1),
                                                      (e.localPosition.dy /
                                                              side)
                                                          .clamp(0, 1),
                                                    ),
                                                  ]);
                                                });
                                              },
                                              onPointerMove: (e) {
                                                if (pointer != e.pointer) {
                                                  return;
                                                }
                                                setState(
                                                  () => strokes.last.add(
                                                    Offset(
                                                      (e.localPosition.dx /
                                                              side)
                                                          .clamp(0, 1),
                                                      (e.localPosition.dy /
                                                              side)
                                                          .clamp(0, 1),
                                                    ),
                                                  ),
                                                );
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
                                              child: ClipRect(
                                                child: CustomPaint(
                                                  painter: InkPainter(
                                                    strokes,
                                                    strokeScale: _strokeScale,
                                                  ),
                                                  child: strokes.isEmpty
                                                      ? const Center(
                                                          child: Text(
                                                            '여기에 쓰세요',
                                                            style: TextStyle(
                                                              fontSize: 15,
                                                              fontWeight:
                                                                  FontWeight
                                                                      .w300,
                                                              color: Color(
                                                                0xff999999,
                                                              ),
                                                            ),
                                                          ),
                                                        )
                                                      : const SizedBox.expand(),
                                                ),
                                              ),
                                            ),
                                    ),
                                  ],
                                ),
                              ),
                            );
                          },
                        ),
                      ),
                      const SizedBox(height: 12),
                      SizedBox(
                        // Reserve the same footprint even when candidates float
                        // above the keyboard, so the writing frame never resizes.
                        height: _candidateRowHeight(context, scale: 1) * 2 + 96,
                        child: MediaQuery.viewInsetsOf(context).bottom > 0
                            ? null
                            : Column(
                                children: [
                                  _candidateList(context),
                                  const SizedBox(height: 8),
                                  _actionRow(),
                                ],
                              ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
          if (MediaQuery.viewInsetsOf(context).bottom > 0)
            Positioned(
              left: 0,
              right: 0,
              bottom:
                  (MediaQuery.viewInsetsOf(context).bottom -
                          (MediaQuery.sizeOf(context).height -
                              bodyConstraints.maxHeight))
                      .clamp(0.0, double.infinity),
              child: Center(
                child: Container(
                  constraints: const BoxConstraints(maxWidth: 600),
                  padding: const EdgeInsets.symmetric(horizontal: 24),
                  color: const Color(0xff191919),
                  child: _candidateList(context),
                ),
              ),
            ),
        ],
      ),
    ),
  );
}
