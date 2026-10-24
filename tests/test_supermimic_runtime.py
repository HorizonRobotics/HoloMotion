"""Optional Isaac lifecycle preserves owned logs and propagates failures."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from holomotion.src.algo.supermimic.runtime import physics_runtime


@pytest.mark.parametrize("fail", [False, True])
def test_isaac_runtime_keeps_log_and_closes_after_body(
    monkeypatch, tmp_path, fail
):
    events = []
    original_argv = sys.argv[:]
    log_path = tmp_path / "isaac_kit.log"
    warp = ModuleType("warp")
    warp.__version__ = "selected"
    warp.init = lambda: events.append("warp")
    warp.get_cuda_device_count = lambda: 1
    app = ModuleType("isaaclab.app")

    def launch(**kwargs):
        assert kwargs == dict(
            headless=True, device="cuda:0", fast_shutdown=False
        )
        assert sys.argv == [original_argv[0], f"--/log/file={log_path}"]
        events.append("launch")
        return SimpleNamespace(
            app=SimpleNamespace(close=lambda **kw: events.append("close"))
        )

    app.AppLauncher = launch
    carb = ModuleType("carb")
    settings = ModuleType("carb.settings")
    settings.get_settings = lambda: SimpleNamespace(
        get=lambda key: str(log_path)
    )
    carb.settings = settings
    for name, module in [
        ("warp", warp),
        ("isaaclab.app", app),
        ("carb", carb),
        ("carb.settings", settings),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    from holomotion.src.env.reference_tracking.isaac_backend import (
        IsaacTrackingBackend,
    )

    monkeypatch.setattr(
        IsaacTrackingBackend,
        "_active",
        SimpleNamespace(close=lambda: events.append("scene_close")),
    )
    options = SimpleNamespace(
        physics_backend="isaac_physx", output=tmp_path, device="cuda:0"
    )

    def run():
        with physics_runtime(options):
            events.append("body")
            if fail:
                raise RuntimeError("training failed")

    if fail:
        with pytest.raises(RuntimeError, match="training failed"):
            run()
    else:
        run()
    assert events == ["launch", "warp", "body", "scene_close", "close"]
    assert sys.argv == original_argv
    assert json.loads((tmp_path / "isaac_logging.json").read_text())[
        "configured_path"
    ] == str(log_path)
