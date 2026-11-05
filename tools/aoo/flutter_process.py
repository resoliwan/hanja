"""Own a Flutter machine-protocol process and await acknowledged operations."""

import json
import os
import queue
import signal
import subprocess
import threading
import time

from build_configuration import dart_define_arguments


class FlutterProcess:
    def __init__(self, app_directory, lifetime_fd, *, web_port=None, device="chrome", hot_restart_url="", hot_restart_token=""):
        self.app_id = None
        self.url = None
        self.ready = False
        self._request_id = 0
        self._closed = False
        self._responses = queue.Queue()
        platform_arguments = (
            ["--web-hostname", "127.0.0.1", "--web-port", str(web_port),
             "--cross-origin-isolation"] if device == "chrome" else []
        )
        self.process = subprocess.Popen(
            ["flutter", "run", "--machine", "-d", device, *platform_arguments,
             *dart_define_arguments(app_directory),
             f"--dart-define=AOO_HOT_RESTART_URL={hot_restart_url}",
             f"--dart-define=AOO_HOT_RESTART_TOKEN={hot_restart_token}"],
            cwd=app_directory, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
            # Keep the session locked even if the supervisor is killed abruptly.
            pass_fds=(lifetime_fd,),
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        for line in self.process.stdout:
            print(line, end="", flush=True)
            try:
                messages = json.loads(line)
            except ValueError:
                continue
            if not isinstance(messages, list):
                continue
            for message in messages:
                if not isinstance(message, dict):
                    continue
                event = message.get("event")
                params = message.get("params", {})
                if event == "app.start":
                    self.app_id = params["appId"]
                elif event == "app.webLaunchUrl":
                    self.url = params["url"]
                elif event == "app.started":
                    self.ready = True
                elif event == "app.stop":
                    self.ready = False
                elif "id" in message:
                    self._responses.put(message)
        self.ready = False
        self._responses.put(None)

    def reload(self, hot):
        if not self.ready or not self.app_id:
            return {"ok": False, "error": "Flutter is not ready. Check session status and log."}
        self._request_id += 1
        request = {"id": self._request_id, "method": "app.restart", "params": {
            "appId": self.app_id, "fullRestart": not hot, "reason": "manual",
        }}
        self.process.stdin.write(json.dumps([request]) + "\n")
        self.process.stdin.flush()
        # The supervisor serves one command at a time, so reloads cannot overlap.
        deadline = time.monotonic() + 180
        while True:
            try:
                response = self._responses.get(timeout=max(0, deadline - time.monotonic()))
            except queue.Empty:
                # Do not accept another reload while an unacknowledged one is running.
                self.close()
                return {"ok": False, "error": "Reload timed out; session stopped. Run ensure again."}
            if response is None:
                return {"ok": False, "error": "Flutter exited during reload. Check the log."}
            if response.get("id") != self._request_id:
                continue
            if "error" in response:
                return {"ok": False, "error": response["error"]}
            result = response.get("result") or {}
            return {"ok": result.get("code", 1) == 0,
                    "operation": "hot-reload" if hot else "hot-restart", **result}

    def close(self):
        if self._closed:
            return
        self._closed = True
        # Signal only the process group we created, never search or kill by port/name.
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=5)
        self._reader.join(timeout=2)
        self.process.stdin.close()
        self.process.stdout.close()
