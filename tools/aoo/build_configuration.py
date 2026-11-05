"""Resolve this checkout's local Flutter integration settings."""

import json
import os
from pathlib import Path


def dart_define_arguments(app_directory):
    app_root = Path(app_directory)
    configured = os.environ.get("HANJA_DART_DEFINES_PATH", "").strip()
    paths = [app_root / configured] if configured else [
        app_root / ".sentry.dart-defines.json",
        app_root / ".analytics.dart-defines.json",
    ]
    arguments = []
    for path in paths:
        if not path.is_file():
            if configured:
                raise RuntimeError(f"Dart defines file not found: {path}")
            continue
        try:
            configuration = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Invalid Dart defines file: {path}") from error
        if not isinstance(configuration, dict):
            raise RuntimeError(f"Dart defines must be a JSON object: {path}")
        arguments.append(f"--dart-define-from-file={path.resolve()}")
    return arguments

