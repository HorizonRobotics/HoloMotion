# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import http.server
import os
import socket
import socketserver
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import webview


# -----------------------------
# UI Shell
# -----------------------------
SHELL_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>SMPL NPZ Viewer</title>
  <style>
    html,body{height:100%;margin:0;background:#1f1f1f;color:#fff;font-family:-apple-system,system-ui}
    .topbar{
      height:44px; display:flex; align-items:center; gap:12px;
      padding:0 12px; background:#2a2a2a; border-bottom:1px solid rgba(255,255,255,.08);
    }
    .btn{
      padding:7px 10px;
      border:1px solid rgba(255,255,255,.25);
      border-radius:10px;
      background:rgba(255,255,255,.08);
      color:#fff;
      cursor:pointer
    }
    .btn:hover{background:rgba(255,255,255,.12)}
    .status{
      font-size:13px;opacity:.85;
      white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
      flex: 1;
    }
    .bar{
      height:8px;width:240px;
      background:rgba(255,255,255,.14);
      border-radius:999px;overflow:hidden;
      display:none
    }
    .bar>div{
      height:100%;width:30%;
      background:rgba(255,255,255,.65);
      border-radius:999px;
      animation:move 1.1s infinite
    }
    @keyframes move{0%{transform:translateX(-120%)}100%{transform:translateX(320%)}}
    .main{height:calc(100% - 44px)}
    iframe{width:100%;height:100%;border:0;background:#000}
  </style>
</head>
<body>
  <div class="topbar">
    <button class="btn" onclick="window.pywebview.api.pick_and_generate()">Load NPZ</button>
    <div class="bar" id="bar"><div></div></div>
    <div class="status" id="status">Select an NPZ file…</div>
  </div>

  <div class="main">
    <iframe id="viewer" src="about:blank"></iframe>
  </div>

  <script>
    function setBusy(b){
      document.getElementById('bar').style.display = b ? 'block' : 'none';
    }
    function setStatus(t){
      document.getElementById('status').textContent = t || '';
    }
    function showViewer(url){
      const u = url + (url.includes('?') ? '&' : '?') + 't=' + Date.now();
      document.getElementById('viewer').src = u;
    }
  </script>
</body>
</html>
"""


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class AppConfig:
    root: Path
    port: int
    smpl_npz_to_html: Path
    template: Path
    out_html: Path
    window_title: str
    width: int
    height: int
    auto_pick: bool
    debug: bool


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SMPL NPZ viewer UI.")
    ap.add_argument("--port", type=int, default=8000, help="Local HTTP port for serving assets.")
    ap.add_argument("--smpl_npz_to_html", type=Path, default=Path("smpl_npz_to_html.py"), help="Path to smpl_npz_to_html.py")
    ap.add_argument("--template", type=Path, default=Path("templates/index_wooden_static.html"), help="HTML template path")
    ap.add_argument("--out", type=Path, default=Path("_generated/vis.html"), help="Output vis.html path")
    ap.add_argument("--title", type=str, default="NPZ Viewer", help="Window title")
    ap.add_argument("--width", type=int, default=800, help="Window width")
    ap.add_argument("--height", type=int, default=600, help="Window height")
    ap.add_argument("--no-auto-pick", action="store_false", help="Do not auto-open file picker at startup")
    ap.add_argument("--debug", action="store_true", help="Enable pywebview debug/devtools")
    return ap.parse_args()


# -----------------------------
# Utilities
# -----------------------------
def js_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def ensure_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")


def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


# -----------------------------
# Core: server + generator + UI API
# -----------------------------
class StaticServer:
    def __init__(self, root: Path, port: int):
        self.root = root
        self.port = port
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        def _serve():
            os.chdir(self.root)  # serve assets from project root
            handler = http.server.SimpleHTTPRequestHandler
            with socketserver.TCPServer(("127.0.0.1", self.port), handler) as httpd:
                httpd.serve_forever()

        self._thread = threading.Thread(target=_serve, daemon=True)
        self._thread.start()


class MakeVisRunner:
    def __init__(self, root: Path, smpl_npz_to_html: Path, template: Path, out_html: Path):
        self.root = root
        self.smpl_npz_to_html = smpl_npz_to_html
        self.template = template
        self.out_html = out_html

    def run(self, npz_path: Path) -> None:
        ensure_exists(self.smpl_npz_to_html, "smpl_npz_to_html.py")
        ensure_exists(self.template, "template html")
        ensure_exists(npz_path, "npz file")

        self.out_html.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(self.smpl_npz_to_html),
            "--npz",
            str(npz_path),
            "--template",
            str(self.template),
            "--out",
            str(self.out_html),
        ]
        subprocess.check_call(cmd, cwd=str(self.root))


def pick_npz_dialog(window) -> Optional[Path]:
    file_types = ("NPZ files (*.npz)", "All files (*.*)")

    # Prefer new enum if available; fallback to deprecated constant.
    try:
        dialog_open = webview.FileDialog.OPEN  # type: ignore[attr-defined]
        paths = window.create_file_dialog(dialog_open, allow_multiple=False, file_types=file_types)
    except Exception:
        paths = window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False, file_types=file_types)

    return Path(paths[0]) if paths else None


class UIAPI:
    def __init__(self, window, cfg: AppConfig, runner: MakeVisRunner):
        self.window = window
        self.cfg = cfg
        self.runner = runner
        self._busy = False

    def pick_and_generate(self) -> None:
        if self._busy:
            return

        npz = pick_npz_dialog(self.window)
        if npz is None:
            return

        safe_name = js_escape(npz.name)
        self.window.evaluate_js(f"setBusy(true); setStatus('Generating: {safe_name}');")

        def worker():
            self._busy = True
            try:
                self.runner.run(npz)
                rel = self.cfg.out_html.relative_to(self.cfg.root).as_posix()
                self.window.evaluate_js(
                    f"setBusy(false); setStatus('Loaded: {safe_name}'); "
                    f"showViewer('http://127.0.0.1:{self.cfg.port}/{rel}');"
                )
            except Exception as e:
                msg = js_escape(str(e))
                self.window.evaluate_js(f"setBusy(false); setStatus('Failed: {msg}');")
            finally:
                self._busy = False

        threading.Thread(target=worker, daemon=True).start()

    def auto_pick_once(self) -> None:
        # Called from window.events.loaded; ensure it runs once.
        if getattr(self, "_auto_done", False):
            return
        setattr(self, "_auto_done", True)
        if self.cfg.auto_pick:
            self.pick_and_generate()


# -----------------------------
# Entrypoint
# -----------------------------
def build_config(args: argparse.Namespace) -> AppConfig:
    root = Path(__file__).resolve().parent
    smpl_npz_to_html = (root / args.smpl_npz_to_html).resolve() if not args.smpl_npz_to_html.is_absolute() else args.smpl_npz_to_html
    template = (root / args.template).resolve() if not args.template.is_absolute() else args.template
    out_html = (root / args.out).resolve() if not args.out.is_absolute() else args.out

    return AppConfig(
        root=root,
        port=int(args.port),
        smpl_npz_to_html=smpl_npz_to_html,
        template=template,
        out_html=out_html,
        window_title=str(args.title),
        width=int(args.width),
        height=int(args.height),
        auto_pick=not bool(args.no_auto_pick),
        debug=bool(args.debug),
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    if not is_port_available(cfg.port):
        raise RuntimeError(f"Port {cfg.port} is already in use. Try --port 8001")

    server = StaticServer(cfg.root, cfg.port)
    server.start()

    runner = MakeVisRunner(cfg.root, cfg.smpl_npz_to_html, cfg.template, cfg.out_html)

    window = webview.create_window(cfg.window_title, html=SHELL_HTML, width=cfg.width, height=cfg.height)
    api = UIAPI(window, cfg, runner)
    window.expose(api.pick_and_generate)

    # Auto pick once on initial load (optional)
    window.events.loaded += lambda: threading.Thread(target=api.auto_pick_once, daemon=True).start()

    webview.start(debug=cfg.debug)


if __name__ == "__main__":
    main()
