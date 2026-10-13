"""Check the controller handoff without importing or contacting a robot SDK."""

import importlib.util
from pathlib import Path

import pytest

script = Path(__file__).resolve().parents[1] / "scripts/control_ownership.py"
spec = importlib.util.spec_from_file_location("control_ownership", script)
ownership = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ownership)


class Client:
    def __init__(self, *, code=0, releases=False):
        self.code = code
        self.name = "ai"
        self.releases = releases
        self.release_calls = 0

    def CheckMode(self):
        return self.code, {"name": self.name}

    def ReleaseMode(self):
        self.release_calls += 1
        if self.releases:
            self.name = ""
        return 0, {}


def test_handoff_confirms_empty_official_mode():
    client = Client(releases=True)
    ownership.acquire_low_level(client, pause=lambda _: None)
    assert client.name == ""
    assert client.release_calls == 1


def test_failed_mode_query_never_releases_control():
    client = Client(code=1)
    with pytest.raises(RuntimeError, match="Cannot confirm"):
        ownership.acquire_low_level(client, pause=lambda _: None)
    assert client.release_calls == 0


def test_failed_handoff_remains_bounded_and_blocks_startup():
    client = Client()
    with pytest.raises(RuntimeError, match="refusing"):
        ownership.acquire_low_level(client, pause=lambda _: None)
    assert client.release_calls == 3
