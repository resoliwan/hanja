"""Per-worktree supervisors reachable across independent shells and agents."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import socketserver
import subprocess
import sys
import time
from urllib.parse import urlsplit

from flutter_process import FlutterProcess
from flutter_targets import FLUTTER_TARGETS
from hot_restart_server import HotRestartServer
from mobile_device import development_host, resolve_device_id


@contextmanager
def _lock(path, *, blocking=True):
    with path.open("a") as handle:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        fcntl.flock(handle, flags)
        try:
            yield handle
        finally:
            # Flutter's detached device helpers can inherit this descriptor.
            # Release explicitly after normal cleanup, even if a helper survives.
            fcntl.flock(handle, fcntl.LOCK_UN)


def _request(directory, action, **options):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(200)
        connection.connect(str(directory / "control.sock"))
        connection.sendall((json.dumps({"action": action, **options}) + "\n").encode())
        with connection.makefile("r") as reader:
            line = reader.readline()
            if not line:
                raise RuntimeError(f"Session supervisor disconnected. Log: {directory / 'server.log'}")
            return json.loads(line)


def _stop(directory):
    try:
        return _request(directory, "stop")
    except (FileNotFoundError, ConnectionRefusedError, ConnectionResetError):
        # Flutter may exit between the status check and the stop request.
        return {"ok": True, "state": "stopped"}


def _status(directory):
    try:
        status = _request(directory, "status")
        # Adopt the active port when upgrading an already-running supervisor.
        port_file = directory / "port"
        if status.get("url") and not port_file.exists():
            port = urlsplit(status["url"]).port
            if port is not None:
                port_file.write_text(str(port))
        return status
    except (FileNotFoundError, ConnectionRefusedError):
        return {"ok": True, "state": "stopped"}


class ServiceSessions:
    def __init__(self, root, *, kind="web"):
        self.kind = kind
        self.root = root.resolve()
        self.registry = Path(os.environ.get(f"AOO_{kind.upper()}_STATE_DIR", Path.home() / ".aoo" / kind))
        self.registry = self.registry.expanduser().resolve()
        self.registry.mkdir(mode=0o700, parents=True, exist_ok=True)
        key = hashlib.sha256(str(self.root).encode()).hexdigest()[:16]
        self.directory = self.registry / key
        self.directory.mkdir(mode=0o700, exist_ok=True)
        if len(os.fsencode(self.directory / "control.sock")) >= 104:
            raise RuntimeError("State directory is too long for a Unix socket; use a shorter path.")

    def execute(self, action, *, hot=False, all_sessions=False):
        if action == "ensure":
            result = self._ensure()
            if result.get("reused"):
                result = {**result, **_request(self.directory, "reload", hot=True)}
            return result
        if action == "list" or (action == "stop" and all_sessions):
            results = []
            for directory in sorted(self.registry.iterdir()):
                if not directory.is_dir():
                    continue
                status = _status(directory)
                if status["state"] == "stopped":
                    continue
                results.append(_stop(directory) if action == "stop" else status)
            return {"ok": all(item.get("ok", True) for item in results), "sessions": results}
        if action == "status":
            return {"worktree": str(self.root), **_status(self.directory)}
        if _status(self.directory)["state"] == "stopped":
            if action == "stop":
                return {"ok": True, "state": "stopped", "worktree": str(self.root)}
            return {"ok": False, "error": f"No server for this worktree. Run aoo {self.kind} ensure."}
        return _stop(self.directory) if action == "stop" else _request(self.directory, action, hot=hot)

    def _ensure(self):
        # A filesystem lock spans shells; a second ensure waits for the first startup.
        with _lock(self.directory / "ensure.lock"):
            status = _status(self.directory)
            if status["state"] == "ready":
                return {**status, "reused": True}
            launcher = None
            log_path = self.directory / "server.log"
            log_offset = log_path.stat().st_size if log_path.exists() else 0
            if status["state"] == "stopped":
                with (self.directory / "server.log").open("a") as log:
                    launcher = subprocess.Popen(
                        [sys.executable, str(Path(__file__).with_name("cli.py")),
                         "_serve", str(self.root), str(self.directory), self.kind],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                        start_new_session=True,
                    )
            deadline = time.monotonic() + FLUTTER_TARGETS[self.kind].startup_timeout
            while time.monotonic() < deadline:
                status = _status(self.directory)
                if status["state"] == "ready":
                    return {**status, "reused": launcher is None}
                if launcher is not None and launcher.poll() is not None:
                    break
                if launcher is None and status["state"] == "stopped":
                    break
                time.sleep(0.15)
            with log_path.open("rb") as log:
                log.seek(max(log_offset, log_path.stat().st_size - 8192))
                lines = log.read().decode("utf-8", errors="replace").splitlines()
            diagnostics = "\n".join(line for line in lines if line.strip() and not line.startswith('[{'))
            detail = f"\n{diagnostics}" if diagnostics else ""
            raise RuntimeError(f"{self.kind} did not become ready. Log: {log_path}{detail}")


def _web_port(directory):
    port_file = directory / "port"
    port = int(port_file.read_text()) if port_file.exists() else 0
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(
                f"Workspace web port {port} is unavailable; stop its owner before retrying."
            ) from error
        port = probe.getsockname()[1]
    port_file.write_text(str(port))
    return port


def serve(root, directory, kind="web"):
    os.umask(0o077)
    # Independent lifetime lock also prevents duplicate supervisors after interruption.
    try:
        with _lock(directory / "supervisor.lock", blocking=False) as lifetime_lock:
            _serve_locked(root, directory, lifetime_lock.fileno(), kind)
    except BlockingIOError:
        raise RuntimeError("A supervisor or its child process still owns this worktree session.")


def _serve_locked(root, directory, lifetime_fd, kind):
    target = FLUTTER_TARGETS[kind]
    socket_path = directory / "control.sock"
    socket_path.unlink(missing_ok=True)
    stopping = False
    process_owner = None
    hot_restart_server = None

    def stop_signal(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            nonlocal stopping, hot_restart_server
            self.connection.settimeout(5)
            try:
                request = json.loads(self.rfile.readline(4096))
                status = {"ok": True, "worktree": str(root), "pid": os.getpid(),
                          "flutterPid": process_owner.process.pid,
                          "url": process_owner.url,
                          "state": "ready" if process_owner.ready and (not target.web or process_owner.url) else "starting",
                          "log": str(directory / "server.log")}
                if request["action"] == "status":
                    result = status
                elif request["action"] == "reload":
                    result = {**status, **process_owner.reload(request.get("hot", False))}
                elif request["action"] == "stop":
                    if hot_restart_server is not None:
                        hot_restart_server.close()
                        hot_restart_server = None
                    stopping = True
                    process_owner.close()
                    result = {**status, "state": "stopped"}
                else:
                    result = {"ok": False, "error": "Unknown supervisor action"}
                self.wfile.write((json.dumps(result) + "\n").encode())
            except (OSError, ValueError, KeyError) as error:
                print(f"Control request failed: {error}", flush=True)

    try:
        with socketserver.UnixStreamServer(str(socket_path), Handler) as server:
            server.timeout = 0.25
            device = resolve_device_id(target.device, physical=True) if target.physical else target.device
            host = development_host() if target.physical else "127.0.0.1"
            web_port = _web_port(directory) if target.web else None
            hot_restart_server = HotRestartServer(
                lambda: _request(directory, "reload", hot=False),
                origin=f"http://127.0.0.1:{web_port}" if web_port else None,
                host=host,
            )
            process_owner = FlutterProcess(
                root / "apps/dic", lifetime_fd, web_port=web_port,
                hot_restart_url=hot_restart_server.url,
                hot_restart_token=hot_restart_server.token,
                device=device,
            )
            startup_deadline = time.monotonic() + target.startup_timeout
            while not stopping and process_owner.process.poll() is None:
                server.handle_request()
                if not process_owner.ready and time.monotonic() > startup_deadline:
                    break
    finally:
        if hot_restart_server is not None:
            hot_restart_server.close()
        if process_owner is not None:
            process_owner.close()
        socket_path.unlink(missing_ok=True)
