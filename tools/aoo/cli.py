"""Manage this checkout's Flutter development sessions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from sessions import ServiceSessions, serve
from flutter_targets import FLUTTER_TARGETS
from build_configuration import dart_define_arguments
from mobile_device import resolve_device_id


def _run_flutter(root, arguments):
    app_directory = root / "apps/dic"
    os.chdir(app_directory)
    if arguments and not arguments[0].startswith("-"):
        query, *arguments = arguments
        target = FLUTTER_TARGETS.get(query)
        if target is None:
            device = resolve_device_id(query)
        elif target.physical:
            device = resolve_device_id(target.device, physical=True)
        else:
            device = target.device
        arguments = ["-d", device, *arguments]
    if not any(arg.split("=", 1)[0] == "--dart-define-from-file" for arg in arguments):
        arguments = [*dart_define_arguments(app_directory), *arguments]
    os.execvp("flutter", ["flutter", "run", *arguments])


def main():
    root = Path(__file__).resolve().parents[2]
    if sys.argv[1:2] == ["run"]:
        _run_flutter(root, sys.argv[2:])
        return
    if sys.argv[1:2] == ["_serve"]:
        serve(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
        return
    parser = argparse.ArgumentParser(prog="aoo")
    targets = parser.add_subparsers(dest="target", required=True)
    targets.add_parser("run", help="flutter run [leo|sm|web|mac|device] [--release|--debug] [Flutter options]")
    for target in FLUTTER_TARGETS:
        actions = targets.add_parser(target).add_subparsers(dest="action", required=True)
        for action in ("run", "ensure", "status", "list"):
            command = actions.add_parser(action)
            if action == "ensure":
                command.add_argument("-r", "--r", dest="action", action="store_const",
                                     const="reload", default="ensure",
                                     help="hot restart the existing session")
        reload = actions.add_parser("reload", help="hot restart; use --hot to preserve state")
        reload.add_argument("--hot", action="store_true")
        stop = actions.add_parser("stop")
        stop.add_argument("--all", action="store_true")
    args = parser.parse_args()
    result = ServiceSessions(root, kind=args.target).execute(
        "ensure" if args.action == "run" else args.action,
        hot=getattr(args, "hot", False), all_sessions=getattr(args, "all", False),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("ok", True):
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, TimeoutError, subprocess.CalledProcessError) as error:
        print(f"aoo: {error}", file=sys.stderr)
        sys.exit(1)
