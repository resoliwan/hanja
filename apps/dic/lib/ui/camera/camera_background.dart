import 'dart:async';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import 'package:image_picker/image_picker.dart';

import '../../adapter/camera/ios_multi_camera.dart';
import '../../dm/model/photo_selection.dart';
import '../../dm/usecase/recognize_photo.dart';
import 'photo_selection_overlay.dart';
import 'capture_file_cleanup_stub.dart'
    if (dart.library.io) 'capture_file_cleanup_io.dart';

/// Controls the camera layer while the search screen owns all navigation and results.
class CameraBackgroundController extends ChangeNotifier {
  _CameraBackgroundState? _state;
  List<String> get candidates => _state?._candidates ?? const [];
  String get message => _state?._message ?? '';
  bool get recognizing => _state?._recognizing ?? false;
  bool get hasPhoto => _state?._photo != null;
  bool get canCapture =>
      _state?._camera != null && !(_state?._capturing ?? true);
  Future<void> capture() async => _state?._capture();
  Future<void> pickPhoto() async => _state?._pickPhoto();
  Future<void> retake() async => _state?._retake();
  void setPreviewFrame(Rect globalFrame) =>
      _state?._setPreviewFrame(globalFrame);
  void clearSelection() => _state?._clearSelection();
  void _publish() => notifyListeners();
}

class CameraBackground extends StatefulWidget {
  const CameraBackground({
    super.key,
    required this.recognizePhoto,
    required this.controller,
  });
  final RecognizePhoto recognizePhoto;
  final CameraBackgroundController controller;
  @override
  State<CameraBackground> createState() => _CameraBackgroundState();
}

class _CameraBackgroundState extends State<CameraBackground>
    with WidgetsBindingObserver {
  Rect? _previewFrame;
  void _setPreviewFrame(Rect globalFrame) {
    final box = context.findRenderObject() as RenderBox?;
    if (box == null) return;
    final local = globalFrame.shift(-box.localToGlobal(Offset.zero));
    if (local == _previewFrame) return;
    setState(() => _previewFrame = local);
  }

  CameraController? _camera;
  Uint8List? _photo;
  Size? _photoSize;
  List<Offset> _points = [];
  PhotoSelection? _selection;
  List<String> _candidates = [];
  bool _capturing = false, _recognizing = false, _starting = false;
  bool _active = true;
  bool _focusing = false;
  int _revision = 0;
  String? _message;

  @override
  void initState() {
    super.initState();
    widget.controller._state = this;
    WidgetsBinding.instance.addObserver(this);
    unawaited(_startCamera());
  }

  Future<void> _startCamera({bool allowMultiCamera = true}) async {
    if (_starting || _photo != null || !_active) return;
    if (!kIsWeb && defaultTargetPlatform == TargetPlatform.macOS) {
      setState(() => _message = 'Mac에서는 사진을 열어 한자를 선택하세요.');
      return;
    }
    _starting = true;
    CameraController? controller;
    bool usingMultiCamera = false;
    try {
      final devices = await availableCameras();
      if (devices.isEmpty) throw StateError('No camera');
      final back = devices
          .where((c) => c.lensDirection == CameraLensDirection.back)
          .toList();
      final fallback = back.firstWhere(
        (c) => c.lensType == CameraLensType.wide,
        orElse: () => back.isEmpty ? devices.first : back.first,
      );
      final multi =
          !kIsWeb &&
              defaultTargetPlatform == TargetPlatform.iOS &&
              allowMultiCamera
          ? await resolveIosMultiCamera()
          : null;
      usingMultiCamera = multi != null;
      final device = multi?.camera ?? fallback;
      controller = CameraController(
        device,
        ResolutionPreset.high,
        enableAudio: false,
      );
      await controller.initialize();
      if (!mounted || _photo != null || !_active) {
        await controller.dispose();
        return;
      }
      if (multi != null) {
        final min = await controller.getMinZoomLevel();
        final max = await controller.getMaxZoomLevel();
        await controller.setZoomLevel(multi.zoom.clamp(min, max));
      }
      await controller.setFocusMode(FocusMode.auto);
      if (controller.value.focusPointSupported) {
        await controller.setFocusPoint(null);
      }
      if (!mounted || !_active || _photo != null) {
        await controller.dispose();
        return;
      }
      setState(() {
        _camera = controller;
        _message = null;
      });
    } on CameraException catch (error) {
      if (_camera == controller) _camera = null;
      await _disposeCamera(controller);
      if (usingMultiCamera && mounted && _active) {
        _starting = false;
        await _startCamera(allowMultiCamera: false);
        return;
      }
      if (mounted) {
        setState(
          () => _message = error.code.contains('Access')
              ? '카메라 권한이 필요합니다. 설정에서 카메라 접근을 허용해 주세요.'
              : '카메라를 열지 못했습니다. 다시 시도해 주세요.',
        );
      }
    } catch (_) {
      if (_camera == controller) _camera = null;
      await _disposeCamera(controller);
      if (usingMultiCamera && mounted && _active) {
        _starting = false;
        await _startCamera(allowMultiCamera: false);
        return;
      }
      if (mounted) setState(() => _message = '카메라를 사용할 수 없습니다. 사진을 열어 주세요.');
    } finally {
      _starting = false;
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    _active = state == AppLifecycleState.resumed;
    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      _revision++;
      final camera = _camera;
      _camera = null;
      unawaited(_disposeCamera(camera));
    } else if (state == AppLifecycleState.resumed &&
        _photo == null &&
        _camera == null) {
      unawaited(_startCamera());
    }
  }

  Future<void> _focusPreview(Offset point, Size viewport, Size preview) async {
    final camera = _camera;
    if (camera == null || _photo != null || !_active || _focusing) return;
    // Cover-fit clips the preview; meter in the full preview's coordinates.
    final normalized = Offset(
      ((point.dx + (preview.width - viewport.width) / 2) / preview.width).clamp(
        0,
        1,
      ),
      ((point.dy + (preview.height - viewport.height) / 2) / preview.height)
          .clamp(0, 1),
    );
    _focusing = true;
    try {
      if (camera.value.exposurePointSupported) {
        await camera.setExposurePoint(normalized);
      }
      if (camera.value.focusPointSupported) {
        await camera.setFocusPoint(normalized);
      }
      if (kDebugMode) {
        debugPrint('Camera focus requested at $normalized');
      }
    } on CameraException catch (error) {
      debugPrint('Camera metering failed: $error');
    } finally {
      _focusing = false;
    }
  }

  Future<void> _disposeCamera(CameraController? camera) async {
    if (camera == null) return;
    await camera.dispose();
  }

  Future<Uint8List> _captureBytes(CameraController camera) async {
    final file = await camera.takePicture();
    try {
      return await file.readAsBytes();
    } finally {
      await deleteCapturedFile(file.path);
    }
  }

  Future<Size> _imageSize(Uint8List bytes) async {
    final codec = await ui.instantiateImageCodec(bytes);
    try {
      final frame = await codec.getNextFrame();
      final size = Size(
        frame.image.width.toDouble(),
        frame.image.height.toDouble(),
      );
      frame.image.dispose();
      return size;
    } finally {
      codec.dispose();
    }
  }

  Future<void> _setPhoto(Uint8List bytes) async {
    final size = await _imageSize(bytes);
    if (!mounted) return;
    _revision++;
    final camera = _camera;
    if (camera != null) {
      await camera.pausePreview();
    }
    if (!mounted) return;
    setState(() {
      _photo = bytes;
      _photoSize = size;
      _points = [];
      _selection = null;
      _candidates = [];
      _message = '인식할 한자를 손가락으로 문질러 주세요';
    });
  }

  Future<void> _capture() async {
    if (_capturing || _camera == null) return;
    final camera = _camera!;
    _revision++;
    setState(() => _capturing = true);
    try {
      await _setPhoto(await _captureBytes(camera));
    } catch (_) {
      if (mounted) {
        setState(() => _message = '촬영하지 못했습니다. 다시 시도해 주세요.');
      }
    } finally {
      if (mounted) setState(() => _capturing = false);
    }
  }

  Future<void> _pickPhoto() async {
    _revision++;
    try {
      final file = await ImagePicker().pickImage(source: ImageSource.gallery);
      if (file != null) await _setPhoto(await file.readAsBytes());
    } catch (_) {
      if (mounted) setState(() => _message = '사진을 열지 못했습니다. 다시 시도해 주세요.');
    }
  }

  Future<void> _retake() async {
    _revision++;
    setState(() {
      _photo = null;
      _photoSize = null;
      _points = [];
      _selection = null;
      _candidates = [];
      _message = null;
    });
    final camera = _camera;
    if (camera != null) {
      await camera.resumePreview();
    } else {
      await _startCamera();
    }
  }

  Future<void> _recognize(
    Uint8List bytes,
    PhotoSelection selection, {
    required int revision,
  }) async {
    if (mounted) setState(() => _recognizing = true);
    try {
      final candidates = await widget.recognizePhoto(bytes, selection);
      if (!mounted || revision != _revision) return;
      setState(() {
        _candidates = candidates;
        _message = candidates.isEmpty ? '인식한 문자가 없습니다. 다시 문질러 주세요.' : null;
      });
    } catch (_) {
      if (mounted && revision == _revision) {
        setState(() {
          _candidates = [];
          _message = '인식하지 못했습니다. 다시 시도해 주세요.';
        });
      }
    } finally {
      if (mounted) setState(() => _recognizing = false);
    }
  }

  void _rub(Offset point, Size size, {bool start = false}) {
    if (_recognizing) return;
    final bounded = Offset(
      point.dx.clamp(0, size.width),
      point.dy.clamp(0, size.height),
    );
    setState(() {
      if (start) {
        _revision++;
        _points = [];
        _candidates = [];
        _message = null;
      }
      _points = [..._points, bounded];
      final radius = PhotoSelectionOverlay.brushWidth / 2;
      final xs = _points.map((p) => p.dx), ys = _points.map((p) => p.dy);
      _selection = PhotoSelection(
        (xs.reduce(math.min) - radius) / size.width,
        (ys.reduce(math.min) - radius) / size.height,
        (xs.reduce(math.max) + radius) / size.width,
        (ys.reduce(math.max) + radius) / size.height,
      );
    });
  }

  void _clearSelection() {
    if (_recognizing) return;
    setState(() {
      _points = [];
      _selection = null;
      _candidates = [];
      _message = '인식할 한자를 손가락으로 문질러 주세요';
    });
  }

  bool _publishScheduled = false;
  @override
  void setState(VoidCallback fn) {
    super.setState(fn);
    if (_publishScheduled) return;
    _publishScheduled = true;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _publishScheduled = false;
      if (mounted) widget.controller._publish();
    });
  }

  @override
  void dispose() {
    widget.controller._state = null;
    WidgetsBinding.instance.removeObserver(this);
    _revision++;
    unawaited(_disposeCamera(_camera));
    super.dispose();
  }

  Widget _photoView() => LayoutBuilder(
    builder: (context, box) {
      final source = _photoSize!;
      final fit = applyBoxFit(BoxFit.contain, source, box.biggest);
      final size = fit.destination;
      return Center(
        child: SizedBox.fromSize(
          size: size,
          child: GestureDetector(
            onPanStart: (d) => _rub(d.localPosition, size, start: true),
            onPanUpdate: (d) => _rub(d.localPosition, size),
            onPanEnd: (_) {
              if (_selection != null && !_recognizing) {
                unawaited(
                  _recognize(_photo!, _selection!, revision: _revision),
                );
              }
            },
            child: Stack(
              fit: StackFit.expand,
              children: [
                Image.memory(_photo!, fit: BoxFit.fill, gaplessPlayback: true),
                ClipRect(
                  child: CustomPaint(
                    painter: PhotoSelectionOverlay(
                      frozen: true,
                      points: _points,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      );
    },
  );

  @override
  Widget build(BuildContext context) {
    if (_photo != null) return _photoView();
    final camera = _camera;
    if (camera == null || !camera.value.isInitialized) {
      return const SizedBox.expand();
    }
    final aspect = MediaQuery.orientationOf(context) == Orientation.portrait
        ? 1 / camera.value.aspectRatio
        : camera.value.aspectRatio;
    return LayoutBuilder(
      builder: (context, box) {
        final scale = math.max(box.maxWidth / aspect, box.maxHeight);
        final preview = Size(aspect * scale, scale);
        return Stack(
          fit: StackFit.expand,
          children: [
            ClipRect(
              child: OverflowBox(
                maxWidth: preview.width,
                maxHeight: preview.height,
                child: SizedBox(
                  width: preview.width,
                  height: preview.height,
                  child: CameraPreview(camera),
                ),
              ),
            ),
            GestureDetector(
              key: const ValueKey('camera-focus-surface'),
              behavior: HitTestBehavior.opaque,
              onTapUp: (details) => unawaited(
                _focusPreview(details.localPosition, box.biggest, preview),
              ),
            ),
            IgnorePointer(
              child: CustomPaint(
                painter: PhotoSelectionOverlay(
                  points: const [],
                  frozen: false,
                  frame: _previewFrame,
                ),
              ),
            ),
          ],
        );
      },
    );
  }
}
