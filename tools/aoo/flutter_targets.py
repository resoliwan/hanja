"""Launch settings shared by the CLI and session supervisor."""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class FlutterTarget:
    device: str
    physical: bool = False
    web: bool = False
    startup_timeout: int = 240


FLUTTER_TARGETS = MappingProxyType({
    "web": FlutterTarget("chrome", web=True),
    "mac": FlutterTarget("macos"),
    "sm": FlutterTarget("sm", physical=True, startup_timeout=900),
    "leo": FlutterTarget("leo", physical=True, startup_timeout=900),
})
