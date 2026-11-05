import 'package:camera/camera.dart';
import 'package:flutter/services.dart';

/// Matches VisionDic's virtual rear camera so iOS can switch lenses for close focus.
Future<({CameraDescription camera, double zoom})?>
resolveIosMultiCamera() async {
  const channel = MethodChannel('hanja/camera-device');
  try {
    final values = await channel.invokeMapMethod<String, Object?>(
      'resolveBackMultiCamera',
    );
    final name = values?['cameraIdentifier'];
    final zoom = values?['initialZoomLevel'];
    if (name is! String || zoom is! num) return null;
    return (
      camera: CameraDescription(
        name: name,
        lensDirection: CameraLensDirection.back,
        sensorOrientation: 90,
        lensType: CameraLensType.wide,
      ),
      zoom: zoom.toDouble(),
    );
  } on MissingPluginException {
    return null;
  } on PlatformException {
    return null;
  }
}
