"""CPU boundary checks for the optional GPU evaluator, with no GPU physics."""

import sys
from types import SimpleNamespace

import mujoco
import pytest
import warp as wp
from mujoco_warp._src import constraint, forward, warp_util

from holomotion.src.evaluation.reference_tracking.canonical_gpu.driver import (
    CanonicalDriver,
    verify_sources,
)
from holomotion.src.evaluation.reference_tracking.evaluation import (
    FullClipEvaluator,
)


def model(cone="pyramidal"):
    # Dispatch-isolation fixtures use the supported geometry count; they
    # stub the full G1 fingerprint and never execute physics.
    geometry = '<geom type="sphere" size=".1" mass="1"/>' * 72
    return mujoco.MjModel.from_xml_string(
        f'<mujoco><option cone="{cone}"/><worldbody><body><freejoint/>{geometry}</body></worldbody></mujoco>'
    )


def test_evaluation_only_guard_precedes_dispatch():
    with pytest.raises(ValueError, match="evaluation-only"):
        CanonicalDriver(model(), evaluation=False)


def test_unsupported_cone_rejected():
    with pytest.raises(ValueError, match="pyramidal"):
        CanonicalDriver(model("elliptic"), evaluation=True)


def test_source_version_mismatch_rejected(monkeypatch):
    import holomotion.src.evaluation.reference_tracking.canonical_gpu.driver as module

    monkeypatch.setattr(
        module.importlib.metadata, "version", lambda package: "other"
    )
    with pytest.raises(RuntimeError, match="requires"):
        verify_sources()


def test_source_digest_mismatch_rejected(monkeypatch):
    import holomotion.src.evaluation.reference_tracking.canonical_gpu.driver as module

    original = module.hashlib.sha256
    monkeypatch.setattr(
        module.hashlib,
        "sha256",
        lambda value: SimpleNamespace(hexdigest=lambda: "tampered"),
    )
    with pytest.raises(RuntimeError, match="source mismatch"):
        verify_sources()
    monkeypatch.setattr(module.hashlib, "sha256", original)


def test_instances_do_not_mutate_native_dispatch_or_share_host_caches(
    monkeypatch,
):
    import holomotion.src.evaluation.reference_tracking.canonical_gpu.driver as driver_module

    monkeypatch.setattr(
        driver_module,
        "verify_model",
        lambda model, expected: "CPU isolation fixture",
    )
    before = (
        wp.launch,
        wp.launch_tiled,
        forward.step,
        constraint.make_constraint,
        wp.config.default_grid_stride,
    )
    modules_before = dict(sys.modules)
    native_cache = dict(warp_util._KERNEL_CACHE)
    a, b = (CanonicalDriver(model(), evaluation=True) for _ in range(2))
    assert (
        wp.launch,
        wp.launch_tiled,
        forward.step,
        constraint.make_constraint,
        wp.config.default_grid_stride,
    ) == before
    assert a.forward is not b.forward and a.forward is not forward
    assert a.constraint is not b.constraint and a.constraint is not constraint
    assert (
        a.host.modules["mujoco_warp._src.warp_util"]._KERNEL_CACHE
        is not b.host.modules["mujoco_warp._src.warp_util"]._KERNEL_CACHE
    )
    a.host.modules["mujoco_warp._src.warp_util"]._KERNEL_CACHE[
        "isolation-check"
    ] = 1
    assert (
        "isolation-check"
        not in b.host.modules["mujoco_warp._src.warp_util"]._KERNEL_CACHE
    )
    assert warp_util._KERNEL_CACHE == native_cache
    assert all(sys.modules[k] is v for k, v in modules_before.items())
    assert a.namespace != b.namespace


def test_unknown_kernel_fails_instead_of_native_fallback(monkeypatch):
    import holomotion.src.evaluation.reference_tracking.canonical_gpu.driver as driver_module

    monkeypatch.setattr(
        driver_module,
        "verify_model",
        lambda model, expected: "CPU isolation fixture",
    )
    a = CanonicalDriver(model(), evaluation=True)
    with pytest.raises(RuntimeError, match="Unqualified canonical kernel"):
        a.launch(SimpleNamespace(key="unqualified"))


def test_unknown_evaluation_mode_rejected_before_gpu_construction():
    with pytest.raises(ValueError, match="Unknown evaluation mode"):
        FullClipEvaluator(
            None, None, None, device="cpu", evaluation_mode="unknown"
        )


def test_fixed_model_guard_rejects_unqualified_model():
    with pytest.raises(ValueError, match="model/options mismatch"):
        CanonicalDriver(model(), evaluation=True)


def test_model_fingerprint_includes_physical_options():
    import hashlib

    from holomotion.src.evaluation.reference_tracking.canonical_gpu.driver import (
        verify_model,
    )

    original = model()
    digest = hashlib.sha256(original.__getstate__()).hexdigest()
    assert verify_model(original, digest) == digest
    original.opt.timestep *= 2
    with pytest.raises(ValueError, match="model/options mismatch"):
        verify_model(original, digest)


def test_unknown_tiled_kernel_rejected_without_native_launch(monkeypatch):
    import holomotion.src.evaluation.reference_tracking.canonical_gpu.driver as driver_module

    monkeypatch.setattr(
        driver_module,
        "verify_model",
        lambda model, expected: "CPU isolation fixture",
    )
    driver = CanonicalDriver(model(), evaluation=True)
    with pytest.raises(
        RuntimeError, match="Unqualified canonical tiled kernel"
    ):
        driver.launch_tiled(SimpleNamespace(key="unqualified"))
