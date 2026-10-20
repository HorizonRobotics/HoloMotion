"""Exercise resumable HTTP transfers against an actual local server."""

import hashlib
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from holoboxing.evaluation.models import _download_file

PAYLOAD = bytes(range(256)) * 1024
EXPECTED = {
    "bytes": len(PAYLOAD),
    "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
}


@contextmanager
def server(*, ignore_range=False, corrupt=False, wrong_range=False):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests.append(self.headers.get("Range"))
            offset = (
                int(self.headers["Range"].split("=")[1].split("-")[0])
                if self.headers.get("Range")
                else 0
            )
            body = b"!" * len(PAYLOAD) if corrupt else PAYLOAD
            self.send_response(200 if ignore_range or not offset else 206)
            if offset and not ignore_range:
                self.send_header(
                    "Content-Range",
                    f"bytes {offset + int(wrong_range)}-{len(body) - 1}/{len(body)}",
                )
                body = body[offset:]
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}/model", requests
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def partial(path):
    return path.with_name(f".{path.name}.{EXPECTED['sha256'][:12]}.partial")


@pytest.mark.parametrize("ignore_range", [False, True])
def test_retained_bytes_resume_or_restart_when_server_ignores_range(
    tmp_path, ignore_range
):
    path = tmp_path / "model.onnx"
    partial(path).write_bytes(PAYLOAD[:12345])
    with server(ignore_range=ignore_range) as (url, requests):
        _download_file(path, url, EXPECTED, retries=0)
    assert path.read_bytes() == PAYLOAD
    assert requests == ["bytes=12345-"]
    assert not partial(path).exists()


def test_failed_transfer_keeps_partial_for_the_next_invocation(tmp_path):
    path = tmp_path / "model.onnx"

    class Interrupt(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD[:12345])
            self.close_connection = True

    http = ThreadingHTTPServer(("127.0.0.1", 0), Interrupt)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(RuntimeError, match="Incomplete download"):
            _download_file(
                path,
                f"http://127.0.0.1:{http.server_port}/model",
                EXPECTED,
                retries=0,
            )
    finally:
        http.shutdown()
        http.server_close()
        thread.join()
    assert partial(path).read_bytes() == PAYLOAD[:12345]
    assert not path.exists()
    with server() as (url, requests):
        _download_file(path, url, EXPECTED, retries=0)
    assert requests == ["bytes=12345-"]
    assert path.read_bytes() == PAYLOAD


def test_invalid_content_range_cannot_append_to_a_valid_prefix(tmp_path):
    path = tmp_path / "model.onnx"
    partial(path).write_bytes(PAYLOAD[:12345])
    with server(wrong_range=True) as (url, _):
        with pytest.raises(RuntimeError, match="Content-Range"):
            _download_file(path, url, EXPECTED, retries=0)
    assert partial(path).read_bytes() == PAYLOAD[:12345]
    assert not path.exists()


def test_wrong_checksum_is_diagnosed_and_quarantined(tmp_path):
    path = tmp_path / "model.onnx"
    with server(corrupt=True) as (url, _):
        with pytest.raises(RuntimeError, match="SHA256 mismatch"):
            _download_file(path, url, EXPECTED, retries=0)
    assert not path.exists()
    assert not partial(path).exists()
    assert partial(path).with_name(
        partial(path).name + ".invalid"
    ).stat().st_size == len(PAYLOAD)
