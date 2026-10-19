"""Serve bundled replay files and explicit assets without importing GPU code."""

from contextlib import contextmanager
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import mimetypes
import ssl
import tempfile
from urllib.parse import unquote, urlsplit

from .resources import ResourceTree


def _file_inside(root, relative):
    root = Path(root).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(relative)
    return path


class ReplayHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, replay, assets, viewer, **kwargs):
        self.replay, self.assets, self.viewer = replay, assets, viewer
        super().__init__(*args, **kwargs)

    def do_HEAD(self):
        self._read(False)

    def do_GET(self):
        self._read(True)

    def _read(self, body):
        route = unquote(urlsplit(self.path).path)
        try:
            if route == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            if route == "/replay.hdf5":
                path = self.replay
            elif route.startswith("/replay/assets/g1-boxing/"):
                path = _file_inside(
                    self.assets,
                    route.removeprefix("/replay/assets/g1-boxing/"),
                )
            elif route.startswith("/assets/g1-boxing/"):
                path = _file_inside(
                    self.assets, route.removeprefix("/assets/g1-boxing/")
                )
            else:
                path = _file_inside(
                    self.viewer,
                    "index.html" if route == "/" else route.lstrip("/"),
                )
            content_type = (
                mimetypes.guess_type(path.name)[0]
                or "application/octet-stream"
            )
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if body:
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        self.wfile.write(chunk)
        except (FileNotFoundError, ValueError, PermissionError):
            self.send_error(404, "Replay resource not found")


@contextmanager
def replay_server(replay, assets, *, host="127.0.0.1", port=8080, viewer=None, tls_cert=None, tls_key=None):
    if bool(tls_cert) != bool(tls_key):
        raise ValueError("HTTPS requires both tls_cert and tls_key")
    tls = None
    if tls_cert:
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(tls_cert, tls_key)
    resources = ResourceTree(assets)
    replay = Path(replay).expanduser().resolve(strict=True)
    viewer = (
        Path(viewer) if viewer else Path(__file__).parent / "viewer" / "dist"
    )
    if not (viewer / "index.html").is_file():
        raise FileNotFoundError(
            "Bundled viewer missing; build holoboxing/evaluation/viewer with npm ci and node build.mjs"
        )
    with tempfile.TemporaryDirectory(prefix="boxing-viewer-") as scratch:
        if replay.suffix.lower() == ".json":
            from .export_hdf5 import export

            converted = Path(scratch) / "replay.hdf5"
            export(
                replay,
                converted,
                asset_manifest=resources.path("manifest.json"),
            )
            replay = converted
        handler = partial(
            ReplayHandler, replay=replay, assets=resources.root, viewer=viewer
        )
        server = ThreadingHTTPServer((host, port), handler)
        try:
            if tls is not None:
                server.socket = tls.wrap_socket(server.socket, server_side=True)
            yield server
        finally:
            server.server_close()


def serve(replay, assets, *, host="127.0.0.1", port=8080, tls_cert=None, tls_key=None):
    with replay_server(replay, assets, host=host, port=port, tls_cert=tls_cert, tls_key=tls_key) as server:
        scheme = "https" if tls_cert else "http"
        print(f"Replay viewer: {scheme}://{host}:{server.server_port}", flush=True)
        server.serve_forever(poll_interval=0.2)
