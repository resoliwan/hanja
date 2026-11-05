import Cocoa
import FlutterMacOS

class MainFlutterWindow: NSWindow {
  override func awakeFromNib() {
    let flutterViewController = FlutterViewController()
    self.contentViewController = flutterViewController
    let phoneSize = NSSize(width: 393, height: 852)
    self.setContentSize(phoneSize)
    self.contentMinSize = phoneSize
    self.contentMaxSize = phoneSize
    self.styleMask.remove(.resizable)
    self.collectionBehavior.insert(.fullScreenNone)
    self.standardWindowButton(.zoomButton)?.isEnabled = false
    self.center()
    self.title = "한자 필기"

    RegisterGeneratedPlugins(registry: flutterViewController)

    super.awakeFromNib()
  }
}
