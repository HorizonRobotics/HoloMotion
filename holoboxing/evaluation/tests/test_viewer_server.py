"""The replay server exposes only selected files and needs no GPU imports."""

from http.client import HTTPConnection
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from holoboxing.evaluation.viewer_server import replay_server
from holoboxing.evaluation.tests.resource_fixtures import make_resources


class ViewerServerTest(unittest.TestCase):
    def test_explicit_files_and_traversal_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets, viewer = root / "assets", root / "viewer"
            make_resources(assets)
            viewer.mkdir()
            (root / "secret").write_bytes(b"private")
            (viewer / "index.html").write_text("<title>Replay</title>")
            replay = root / "replay.hdf5"
            replay.write_bytes(b"selected replay")
            (assets / "outside").symlink_to(root / "secret")
            with replay_server(
                replay, assets, viewer=viewer, port=0
            ) as server:
                thread = threading.Thread(
                    target=server.serve_forever, daemon=True
                )
                thread.start()
                try:
                    for path, status, body in (
                        ("/", 200, b"<title>Replay</title>"),
                        ("/replay.hdf5", 200, b"selected replay"),
                        (
                            "/replay/assets/g1-boxing/scene.xml",
                            200,
                            (assets / "scene.xml").read_bytes(),
                        ),
                        ("/favicon.ico", 204, b""),
                        ("/%2e%2e/secret", 404, None),
                        ("/replay/assets/g1-boxing/%2e%2e/secret", 404, None),
                        ("/replay/assets/g1-boxing/outside", 404, None),
                    ):
                        with self.subTest(path=path):
                            connection = HTTPConnection(
                                "127.0.0.1", server.server_port, timeout=5
                            )
                            connection.request("GET", path)
                            response = connection.getresponse()
                            self.assertEqual(response.status, status)
                            content = response.read()
                            if body is not None:
                                self.assertEqual(content, body)
                            self.assertNotIn(b"private", content)
                            connection.close()
                finally:
                    server.shutdown()
                    thread.join(timeout=5)

    def test_partial_tls_configuration_is_rejected(self):
        for options in (dict(tls_cert="cert.pem"), dict(tls_key="key.pem")):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, "both"),
            ):
                with replay_server("unused", "unused", **options):
                    self.fail("Partial TLS configuration was accepted")

    def test_server_import_does_not_load_gpu_or_training_packages(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import holoboxing.evaluation.viewer_server, sys; "
                "assert not {'torch', 'onnxruntime', 'mujoco', 'warp', 'isaaclab'} & sys.modules.keys()",
            ],
            check=True,
        )
