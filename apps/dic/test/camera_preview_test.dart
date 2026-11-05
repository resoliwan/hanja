import 'dart:async';
import 'dart:math' show Point;

import 'package:camera_platform_interface/camera_platform_interface.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:hanja_app/dm/model/photo_selection.dart';
import 'package:hanja_app/dm/usecase/recognize_photo.dart';
import 'package:hanja_app/out/photo_recognizer.dart';
import 'package:hanja_app/ui/camera/camera_background.dart';
import 'package:hanja_app/ui/hanja_app.dart';
import 'package:hanja_app/dm/model/handwriting.dart';
import 'package:hanja_app/dm/usecase/recognize_handwriting.dart';
import 'package:hanja_app/dm/usecase/open_hanja_dictionary.dart';
import 'package:hanja_app/out/handwriting_recognizer.dart';
import 'package:hanja_app/out/external_dictionary.dart';

void main() {
  testWidgets('Camera only recognizes after manual capture and rubbing', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(400, 600);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.reset);
    final previous = CameraPlatform.instance;
    final camera = _VideoCamera();
    CameraPlatform.instance = camera;
    debugDefaultTargetPlatformOverride = TargetPlatform.android;

    final recognizer = _Recognizer();
    try {
      await tester.pumpWidget(
        HanjaApp(
          recognizePhoto: RecognizePhoto(recognizer),
          recognizeHandwriting: RecognizeHandwriting(_HandwritingRecognizer()),
          openDictionary: OpenHanjaDictionary(_Dictionary()),
        ),
      );
      await tester.pumpAndSettle();
      await tester.tap(find.text('카메라'));
      await tester.pumpAndSettle();
      final controller = tester
          .widget<CameraBackground>(find.byType(CameraBackground))
          .controller;
      await tester.pump();
      await tester.pump(const Duration(seconds: 5));
      expect(camera.frames.hasListener, isFalse);
      expect(recognizer.calls, 0);
      expect(camera.focusMode, FocusMode.auto);
      expect(camera.focusPoints, [null]);
      final focusRect = tester.getRect(
        find.byKey(const ValueKey('camera-focus-surface')),
      );
      await tester.tapAt(
        focusRect.topLeft +
            Offset(focusRect.width * .25, focusRect.height * .25),
      );
      await tester.pumpAndSettle();
      expect(camera.focusPoints.last!.x, closeTo(125 / 450, .001));
      expect(camera.focusPoints.last!.y, closeTo(.25, .001));
      expect(camera.exposurePoint, camera.focusPoints.last);
      expect(camera.photographs, 0);
      expect(recognizer.calls, 0);
      await tester.runAsync(controller.capture);
      await tester.pumpAndSettle();
      expect(camera.photographs, 1);
      expect(controller.hasPhoto, isTrue);
      expect(recognizer.calls, 0);
      await tester.dragFrom(
        tester.getCenter(find.byType(Image)),
        const Offset(30, 20),
      );
      await tester.pumpAndSettle();
      expect(recognizer.calls, 1);
      expect(controller.candidates, ['天']);
      await controller.retake();
      await tester.pumpAndSettle();
      await tester.pump(const Duration(seconds: 5));
      expect(camera.photographs, 1);
      expect(recognizer.calls, 1);
      expect(camera.frames.hasListener, isFalse);
      await tester.pumpWidget(const SizedBox());
      await tester.pumpAndSettle();
      expect(camera.frames.hasListener, isFalse);
    } finally {
      CameraPlatform.instance = previous;
      debugDefaultTargetPlatformOverride = null;
    }
  });
}

class _Recognizer implements PhotoRecognizer {
  int calls = 0;
  @override
  Future<List<String>> recognize(
    Uint8List image,
    PhotoSelection selection,
  ) async {
    calls++;
    expect(img.decodePng(image), isNotNull);
    return ['天'];
  }
}

class _VideoCamera extends CameraPlatform {
  final frames = StreamController<CameraImageData>.broadcast();
  final errors = StreamController<CameraErrorEvent>.broadcast();
  int photographs = 0;
  FocusMode? focusMode;
  final focusPoints = <Point<double>?>[];
  Point<double>? exposurePoint;
  @override
  Future<void> setFocusMode(int cameraId, FocusMode mode) async {
    focusMode = mode;
  }

  @override
  Future<void> setFocusPoint(int cameraId, Point<double>? point) async {
    focusPoints.add(point);
  }

  @override
  Future<void> setExposurePoint(int cameraId, Point<double>? point) async {
    exposurePoint = point;
  }

  @override
  Future<List<CameraDescription>> availableCameras() async => [
    const CameraDescription(
      name: 'test',
      lensDirection: CameraLensDirection.back,
      sensorOrientation: 90,
    ),
  ];
  @override
  Future<int> createCameraWithSettings(
    CameraDescription description,
    MediaSettings settings,
  ) async => 1;
  @override
  Future<void> initializeCamera(
    int cameraId, {
    ImageFormatGroup imageFormatGroup = ImageFormatGroup.unknown,
  }) async {}
  @override
  Stream<CameraInitializedEvent> onCameraInitialized(int cameraId) =>
      Stream.value(
        const CameraInitializedEvent(
          1,
          640,
          480,
          ExposureMode.auto,
          true,
          FocusMode.auto,
          true,
        ),
      );
  @override
  Stream<CameraErrorEvent> onCameraError(int cameraId) => errors.stream;
  @override
  Stream<DeviceOrientationChangedEvent> onDeviceOrientationChanged() =>
      const Stream.empty();
  @override
  bool supportsImageStreaming() => true;
  @override
  Stream<CameraImageData> onStreamedFrameAvailable(
    int cameraId, {
    CameraImageStreamOptions? options,
  }) => frames.stream;
  @override
  Widget buildPreview(int cameraId) => const SizedBox.expand();
  @override
  Future<XFile> takePicture(int cameraId) async {
    photographs++;
    return XFile.fromData(
      Uint8List.fromList(img.encodePng(img.Image(width: 100, height: 100))),
    );
  }

  @override
  Future<void> pausePreview(int cameraId) async {}
  @override
  Future<void> resumePreview(int cameraId) async {}
  @override
  Future<void> dispose(int cameraId) async {}
}

class _HandwritingRecognizer implements HandwritingRecognizer {
  @override
  Future<void> load() async {}
  @override
  Future<List<String>> predict(Handwriting handwriting) async => [];
  @override
  Future<void> dispose() async {}
}

class _Dictionary implements ExternalDictionary {
  @override
  Future<void> open(String character) async {}
}
