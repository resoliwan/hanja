"""Authenticated HTTP bridge to one session's Flutter control channel."""

import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class HotRestartServer:
    def __init__(self, reload, *, origin=None, host="127.0.0.1"):
        self.token = secrets.token_urlsafe(32)
        token = self.token

        class Handler(BaseHTTPRequestHandler):
            def _allowed_origin(self):
                return self.headers.get("Origin") in (None, origin)

            def _respond(self, status, result=None):
                self.send_response(status)
                if origin and self.headers.get("Origin") == origin:
                    self.send_header("Access-Control-Allow-Origin", origin)
                    self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Authorization")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                if result is not None:
                    self.wfile.write(json.dumps(result).encode())

            def do_OPTIONS(self):
                self._respond(204 if self._allowed_origin() else 403)

            def do_POST(self):
                if self.path != "/reload":
                    self._respond(404)
                    return
                if not self._allowed_origin() or not secrets.compare_digest(
                    self.headers.get("Authorization", ""), f"Bearer {token}"
                ):
                    self._respond(403)
                    return
                try:
                    result = reload()
                except (OSError, ValueError, RuntimeError) as error:
                    result = {"ok": False, "error": str(error)}
                self._respond(200 if result.get("ok") else 503, result)

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer((host, 0), Handler)
        self.url = f"http://{host}:{self._server.server_port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()
