"""Loopback-only HTTP server for the interactive browser interface."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from eeg_speller.web.engine import SessionManager, available_configs, gui_symbols


STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 65536


def make_handler(manager: SessionManager, qwen=None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "EEGSpellerLocal/0.1"

        def _send(self, status, body, content_type="application/json; charset=utf-8"):
            data = (json.dumps(body, ensure_ascii=False).encode("utf-8")
                    if content_type.startswith("application/json") else body)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; script-src 'self'; style-src 'self'; "
                             "img-src 'self' data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self._valid_host():
                return self._send(403, {"error": "loopback host required"})
            url = urlsplit(self.path)
            if url.path == "/api/configs":
                return self._send(200, {"configs": available_configs(), "symbols": gui_symbols(),
                                        "qwen": qwen.status() if qwen else {"ready": False,
                                                                              "message": "model not configured"}})
            if url.path == "/api/state":
                try:
                    session_id = parse_qs(url.query).get("session_id", [""])[0]
                    return self._send(200, manager.get(session_id).snapshot())
                except ValueError as exc:
                    return self._send(404, {"error": str(exc)})
            files = {"/": ("index.html", "text/html; charset=utf-8"),
                     "/app.js": ("app.js", "application/javascript; charset=utf-8"),
                     "/style.css": ("style.css", "text/css; charset=utf-8")}
            item = files.get(url.path)
            if item is None:
                return self._send(404, {"error": "not found"})
            name, mime = item
            return self._send(200, (STATIC / name).read_bytes(), mime)

        def do_POST(self):
            if not self._valid_host():
                return self._send(403, {"error": "loopback host required"})
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            if origin and origin not in {"http://" + host, "https://" + host}:
                return self._send(403, {"error": "cross-origin request rejected"})
            if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                return self._send(415, {"error": "JSON required"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY:
                    raise ValueError("invalid request size")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("object required")
                if self.path == "/api/start":
                    result = manager.start(body.get("config", ""), body.get("target", ""))
                elif self.path == "/api/epoch":
                    result = manager.get(body.get("session_id", "")).submit(body.get("hits", []))
                elif self.path == "/api/stop":
                    result = manager.get(body.get("session_id", "")).stop()
                elif self.path == "/api/suggest":
                    if qwen is None:
                        raise ValueError("Qwen is not configured")
                    session = manager.get(body.get("session_id", ""))
                    result = {"suggestion": qwen.suggest(session.primary.value)}
                else:
                    return self._send(404, {"error": "not found"})
                return self._send(200, result)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                return self._send(400, {"error": str(exc)})
            except Exception as exc:
                self.log_error("request failed: %s", exc)
                return self._send(500, {"error": "server error; see local terminal"})

        def _valid_host(self):
            host = self.headers.get("Host", "").split(":", 1)[0].lower()
            return host in {"127.0.0.1", "localhost"}

        def log_message(self, fmt, *args):
            # Avoid writing target text or session contents into access logs.
            if not self.path.startswith("/api/"):
                super().log_message(fmt, *args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Local-only EEG speller browser interface")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=Path("runs/gui"))
    parser.add_argument("--qwen-model", type=Path, help="existing local GGUF model file")
    parser.add_argument("--qwen-cli", type=Path, help="CPU-only llama.cpp executable")
    args = parser.parse_args()
    if not 0 < args.port < 65536:
        parser.error("invalid port")
    qwen = None
    if args.qwen_model:
        from eeg_speller.web.qwen import LocalQwen
        if not args.qwen_cli:
            parser.error("--qwen-cli is required with --qwen-model")
        qwen = LocalQwen(args.qwen_model, args.qwen_cli)
    manager = SessionManager(args.output)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(manager, qwen))
    print(f"Open http://127.0.0.1:{args.port}/ — local-only GUI", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
