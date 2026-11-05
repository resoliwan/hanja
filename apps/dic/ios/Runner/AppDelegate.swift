import Flutter
import UIKit
import AVFoundation

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)
    if let registrar = engineBridge.pluginRegistry.registrar(forPlugin: "HanjaCameraDevice") {
      let channel = FlutterMethodChannel(name: "hanja/camera-device", binaryMessenger: registrar.messenger())
      channel.setMethodCallHandler { call, result in
        guard call.method == "resolveBackMultiCamera" else {
          result(FlutterMethodNotImplemented)
          return
        }
        let device = AVCaptureDevice.default(.builtInTripleCamera, for: .video, position: .back)
          ?? AVCaptureDevice.default(.builtInDualWideCamera, for: .video, position: .back)
        guard let device else { result(nil); return }
        result([
          "cameraIdentifier": device.uniqueID,
          "initialZoomLevel": device.virtualDeviceSwitchOverVideoZoomFactors.first?.doubleValue ?? 1,
        ])
      }
    }
  }
}
