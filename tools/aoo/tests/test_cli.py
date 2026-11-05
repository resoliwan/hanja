"""Exercise detached supervisors with an executable Flutter protocol double."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import socket
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import subprocess
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1]
FAKE_FLUTTER = '''#!/usr/bin/env python3
import json, os, socket, sys, time
from pathlib import Path
root = Path.cwd()
(root / "arguments").write_text(json.dumps(sys.argv[1:]))
if sys.argv[1:] == ["devices", "--machine"]:
    print((root / "devices.json").read_text())
    sys.exit(0)
if "--machine" not in sys.argv:
    print(json.dumps({"args": sys.argv[1:], "cwd": str(root)}))
    sys.exit(0)
with (root / "starts").open("a") as f:
    f.write("started\\n")
if (root / "fail-start").exists():
    print((root / "fail-start").read_text(), flush=True)
    sys.exit(1)
server = socket.socket()
server.bind(("127.0.0.1", int(sys.argv[sys.argv.index("--web-port") + 1]) if "--web-port" in sys.argv else 0))
server.listen()
def emit(message):
    print(json.dumps([message]), flush=True)
time.sleep(0.2)
emit({"event": "app.start", "params": {"appId": "test"}})
if "--web-port" in sys.argv:
    emit({"event": "app.webLaunchUrl", "params": {"url": "http://127.0.0.1:" + str(server.getsockname()[1])}})
emit({"event": "app.started", "params": {"appId": "test"}})
for line in sys.stdin:
    request = json.loads(line)[0]
    with (root / "requests").open("a") as f:
        f.write(json.dumps(request) + "\\n")
    if (root / "exit-reload").exists():
        sys.exit(1)
    time.sleep(0.05)
    code = 1 if (root / "fail-reload").exists() else 0
    emit({"id": request["id"], "result": {"code": code, "message": "compile error" if code else "done"}})
if (root / "linger-on-eof").exists():
    time.sleep(60)
'''

class FlutterCliTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aoo-", dir="/tmp")
        self.base = Path(self.temp.name).resolve()
        self.env = {**os.environ, "AOO_WEB_STATE_DIR": str(self.base / "web"),
                    "AOO_MAC_STATE_DIR": str(self.base / "mac"),
                    "AOO_SM_STATE_DIR": str(self.base / "sm"),
                    "AOO_LEO_STATE_DIR": str(self.base / "leo"),
                    "AOO_DEV_HOST": "127.0.0.1",
                    "PATH": str(self.base / "bin") + os.pathsep + os.environ["PATH"]}
        (self.base / "bin").mkdir()
        flutter = self.base / "bin/flutter"
        flutter.write_text(FAKE_FLUTTER)
        flutter.chmod(0o755)
        self.a = self.make_root("a")
        self.b = self.make_root("b")

    def tearDown(self):
        for target in ("web", "mac", "sm", "leo"):
            self.run_cli(self.a, target, "stop", "--all")
        self.temp.cleanup()

    def make_root(self, name):
        root = self.base / name
        (root / "apps/dic").mkdir(parents=True)
        (root / "apps/dic/devices.json").write_text(json.dumps([
            {"id": "android-phone", "name": "SM-S926N", "targetPlatform": "android-arm64", "emulator": False},
            {"id": "ios-phone", "name": "Leo’s iPhone", "targetPlatform": "ios", "emulator": False},
            {"id": "ios-simulator", "name": "Leo simulator", "targetPlatform": "ios", "emulator": True},
        ]))
        shutil.copytree(SOURCE, root / "tools/aoo", ignore=shutil.ignore_patterns("tests", "__pycache__"))
        return root

    def run_cli(self, root, *args, success=True):
        result = subprocess.run(
            ["python3", str(root / "tools/aoo/cli.py"), *args],
            env=self.env, cwd=root / "apps/dic", capture_output=True, text=True, timeout=15,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return json.loads(result.stdout) if result.stdout else result.stderr

    def test_run_mobile_forwards_modes_and_uses_app_directory(self):
        configuration = self.a / "apps/dic/.sentry.dart-defines.json"
        configuration.write_text('{"SENTRY_DSN": "test"}')
        for target, device in (("leo", "ios-phone"), ("sm", "android-phone")):
            for mode in ("--release", "--debug"):
                result = self.run_cli(self.a, "run", target, mode, "--dart-define=EXAMPLE=true")
                self.assertEqual(result["cwd"], str(self.a / "apps/dic"))
                self.assertEqual(result["args"], [
                    "run", f"--dart-define-from-file={configuration}",
                    "-d", device, mode, "--dart-define=EXAMPLE=true",
                ])

    def test_run_respects_explicit_configuration_and_default_mode(self):
        (self.a / "apps/dic/.sentry.dart-defines.json").write_text('{}')
        for option in (["--dart-define-from-file=custom.json"],
                       ["--dart-define-from-file", "custom.json"]):
            result = self.run_cli(self.a, "run", "sm", *option)
            self.assertEqual(result["args"], ["run", "-d", "android-phone", *option])

    def test_run_missing_device_fails_before_launch(self):
        result = self.run_cli(self.a, "run", "missing-phone", "--release", success=False)
        self.assertIn("no connected device matches", result)
        arguments = json.loads((self.a / "apps/dic/arguments").read_text())
        self.assertEqual(arguments, ["devices", "--machine"])

    def test_http_hot_restart_routes_to_all_target_sessions(self):
        for target in ("web", "mac", "sm", "leo"):
            with self.subTest(target=target):
                status = self.run_cli(self.a, target, "ensure")
                arguments = json.loads((self.a / "apps/dic/arguments").read_text())
                defines = dict(arg.removeprefix("--dart-define=").split("=", 1)
                               for arg in arguments if arg.startswith("--dart-define="))
                url = defines["AOO_HOT_RESTART_URL"] + "/reload"
                headers = {"Authorization": "Bearer " + defines["AOO_HOT_RESTART_TOKEN"]}
                if target == "web":
                    headers["Origin"] = status["url"]
                    with urlopen(Request(url, method="OPTIONS", headers={"Origin": status["url"]})) as response:
                        self.assertEqual(response.headers["Access-Control-Allow-Origin"], status["url"])
                for bad_headers in ({}, {**headers, "Origin": "https://example.com"}):
                    with self.assertRaises(HTTPError) as failure:
                        urlopen(Request(url, method="POST", headers=bad_headers))
                    self.assertEqual(failure.exception.code, 403)
                    failure.exception.close()
                with urlopen(Request(url, method="POST", headers=headers)) as response:
                    self.assertEqual(json.load(response)["operation"], "hot-restart")
                requests = (self.a / "apps/dic/requests").read_text().splitlines()
                self.assertTrue(json.loads(requests[-1])["params"]["fullRestart"])
                marker = self.a / "apps/dic/fail-reload"
                marker.touch()
                with self.assertRaises(HTTPError) as failure:
                    urlopen(Request(url, method="POST", headers=headers))
                self.assertEqual(failure.exception.code, 503)
                failure.exception.close()
                marker.unlink()
                self.run_cli(self.a, target, "stop")

    def test_ensure_hot_reloads_running_flutter_for_each_target(self):
        for target in ("web", "mac", "sm", "leo"):
            with self.subTest(target=target):
                first = self.run_cli(self.a, target, "run")
                self.assertFalse(first["reused"])
                again = self.run_cli(self.a, target, "ensure")
                self.assertTrue(again["reused"])
                self.assertEqual(again["flutterPid"], first["flutterPid"])
                self.assertEqual(again["operation"], "hot-reload")
                request = json.loads((self.a / "apps/dic/requests").read_text().splitlines()[-1])
                self.assertFalse(request["params"]["fullRestart"])
                marker = self.a / "apps/dic/fail-reload"
                marker.touch()
                failed = self.run_cli(self.a, target, "ensure", success=False)
                self.assertEqual(failed["message"], "compile error")
                marker.unlink()
                self.assertTrue(self.run_cli(self.a, target, "ensure")["ok"])

    def test_mobile_targets_resolve_physical_devices_and_keep_sessions_separate(self):
        for target, device in (("sm", "android-phone"), ("leo", "ios-phone")):
            first = self.run_cli(self.a, target, "ensure")
            self.assertEqual(first["state"], "ready")
            self.assertIsNone(first["url"])
            arguments = json.loads((self.a / "apps/dic/arguments").read_text())
            self.assertEqual(arguments[arguments.index("-d") + 1], device)
            self.assertNotIn("--web-port", arguments)
            second = self.run_cli(self.b, target, "ensure")
            self.assertNotEqual(first["flutterPid"], second["flutterPid"])
            self.run_cli(self.a, target, "stop")
            self.assertEqual(self.run_cli(self.b, target, "status")["state"], "ready")

    def test_ensure_r_restarts_existing_sessions_only(self):
        for target in ("web", "mac", "sm", "leo"):
            self.run_cli(self.a, target, "ensure", "-r", success=False)
            first = self.run_cli(self.a, target, "ensure")
            for flag in ("-r", "--r"):
                result = self.run_cli(self.a, target, "ensure", flag)
                self.assertEqual(result["operation"], "hot-restart")
                self.assertEqual(result["flutterPid"], first["flutterPid"])

    def test_missing_or_ambiguous_mobile_device_does_not_launch(self):
        device_file = self.a / "apps/dic/devices.json"
        for devices in ([], [
            {"id": "one", "name": "SM-one", "targetPlatform": "android-arm64"},
            {"id": "two", "name": "SM-two", "targetPlatform": "android-arm64"},
        ]):
            device_file.write_text(json.dumps(devices))
            self.run_cli(self.a, "sm", "ensure", success=False)
            self.assertFalse((self.a / "apps/dic/starts").exists())

    def test_mac_is_ready_without_web_url_and_uses_native_arguments(self):
        first = self.run_cli(self.a, "mac", "ensure")
        self.assertEqual(first["state"], "ready")
        self.assertIsNone(first["url"])
        arguments = json.loads((self.a / "apps/dic/arguments").read_text())
        self.assertEqual(arguments[arguments.index("-d") + 1], "macos")
        self.assertNotIn("--web-port", arguments)
        self.assertNotIn("--cross-origin-isolation", arguments)
        second = self.run_cli(self.b, "mac", "ensure")
        self.assertNotEqual(first["flutterPid"], second["flutterPid"])
        self.run_cli(self.a, "mac", "reload")
        request = json.loads((self.a / "apps/dic/requests").read_text().splitlines()[-1])
        self.assertTrue(request["params"]["fullRestart"])
        self.run_cli(self.a, "mac", "stop")
        self.assertEqual(self.run_cli(self.b, "mac", "status")["state"], "ready")

    def test_web_keeps_its_port_after_stop_and_restart(self):
        first = self.run_cli(self.a, "web", "ensure")
        self.run_cli(self.a, "web", "stop")
        second = self.run_cli(self.a, "web", "ensure")
        self.assertEqual(first["url"], second["url"])
        self.assertNotEqual(first["flutterPid"], second["flutterPid"])
        args = json.loads((self.a / "apps/dic/arguments").read_text())
        self.assertEqual(args[args.index("--web-port") + 1], str(urlsplit(first["url"]).port))
        self.assertIn("--cross-origin-isolation", args)

    def test_busy_workspace_port_fails_without_switching_ports(self):
        first = self.run_cli(self.a, "web", "ensure")
        self.run_cli(self.a, "web", "stop")
        with socket.socket() as owner:
            owner.bind(("127.0.0.1", urlsplit(first["url"]).port))
            owner.listen()
            self.run_cli(self.a, "web", "ensure", success=False)
        resumed = self.run_cli(self.a, "web", "ensure")
        self.assertEqual(resumed["url"], first["url"])

    def test_concurrent_ensure_reuses_one_process_and_url(self):
        with ThreadPoolExecutor(max_workers=4) as workers:
            results = list(workers.map(lambda _: self.run_cli(self.a, "web", "ensure"), range(4)))
        self.assertEqual(len({r["pid"] for r in results}), 1)
        self.assertEqual(len({r["url"] for r in results}), 1)
        self.assertEqual((self.a / "apps/dic/starts").read_text(), "started\n")
        self.assertEqual(sum(r.get("operation") == "hot-reload" for r in results), 3)
        requests = [json.loads(line) for line in (self.a / "apps/dic/requests").read_text().splitlines()]
        self.assertEqual([r["params"]["fullRestart"] for r in requests], [False] * 3)

    def test_worktrees_are_isolated_and_reload_defaults_to_restart(self):
        a = self.run_cli(self.a, "web", "ensure")
        b = self.run_cli(self.b, "web", "ensure")
        self.assertNotEqual(a["url"], b["url"])
        self.run_cli(self.b, "web", "reload")
        self.run_cli(self.b, "web", "reload", "--hot")
        requests = [json.loads(line) for line in (self.b / "apps/dic/requests").read_text().splitlines()]
        self.assertEqual([r["params"]["fullRestart"] for r in requests], [True, False])
        self.assertFalse((self.a / "apps/dic/requests").exists())
        self.assertEqual(len(self.run_cli(self.a, "web", "list")["sessions"]), 2)
        self.run_cli(self.a, "web", "stop")
        self.assertEqual(self.run_cli(self.b, "web", "status")["state"], "ready")
        self.assertEqual(self.run_cli(self.a, "web", "status")["state"], "stopped")

    def test_reload_failure_is_reported_and_next_reload_can_recover(self):
        self.run_cli(self.a, "web", "ensure")
        marker = self.a / "apps/dic/fail-reload"
        marker.touch()
        failed = self.run_cli(self.a, "web", "reload", success=False)
        self.assertEqual(failed["message"], "compile error")
        marker.unlink()
        self.assertTrue(self.run_cli(self.a, "web", "reload")["ok"])

    def test_start_failure_and_missing_server_are_errors(self):
        self.run_cli(self.a, "web", "reload", success=False)
        marker = self.a / "apps/dic/fail-start"
        marker.write_text("Error connecting to the service protocol: Connection reset by peer")
        error = self.run_cli(self.a, "web", "ensure", success=False)
        self.assertIn("server.log", error)
        self.assertIn(marker.read_text(), error)
        marker.write_text("A different startup failure")
        error = self.run_cli(self.a, "web", "ensure", success=False)
        self.assertIn(marker.read_text(), error)
        self.assertNotIn("Connection reset by peer", error)

    def test_flutter_exit_during_reload_is_reported(self):
        self.run_cli(self.a, "web", "ensure")
        (self.a / "apps/dic/exit-reload").touch()
        result = self.run_cli(self.a, "web", "reload", success=False)
        self.assertIn("exited", result["error"])

    def test_parallel_reloads_receive_their_own_acknowledgements(self):
        self.run_cli(self.a, "web", "ensure")
        with ThreadPoolExecutor(max_workers=3) as workers:
            results = list(workers.map(
                lambda hot: self.run_cli(self.a, "web", "reload", *(["--hot"] if hot else [])),
                [True, False, True],
            ))
        self.assertEqual([r["operation"] for r in results],
                         ["hot-reload", "hot-restart", "hot-reload"])

