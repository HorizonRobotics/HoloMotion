"""Torch bindings accept device descriptors from either observation backend."""

from types import SimpleNamespace

import pytest
import torch

from holoboxing.evaluation.runtime import onnx_device
from holoboxing.training.envs.onnx_lower import TrainingBoundSession


@pytest.mark.parametrize("compiled", [False, True])
def test_bound_session_normalizes_device_before_allocating_and_synchronizing(
    monkeypatch, compiled
):
    descriptor = (
        onnx_device.CudaDevice(3) if compiled else torch.device("cuda:3")
    )
    monkeypatch.setattr(onnx_device, "require_gpu", lambda _: descriptor)
    allocated = []
    original_zeros, original_empty = torch.zeros, torch.empty

    def allocate(function, shape, *, device, dtype):
        assert isinstance(device, torch.device)
        assert device.type == "cuda" and device.index == 3
        allocated.append(device)
        # Test transport only; the separate GPU smoke exercises real storage.
        return function(shape, dtype=dtype)

    monkeypatch.setattr(
        torch,
        "zeros",
        lambda shape, **kw: allocate(original_zeros, shape, **kw),
    )
    monkeypatch.setattr(
        torch,
        "empty",
        lambda shape, **kw: allocate(original_empty, shape, **kw),
    )
    binding = SimpleNamespace(
        bind_input=lambda *a: None, bind_output=lambda *a: None
    )
    session = SimpleNamespace(
        get_providers=lambda: ["CUDAExecutionProvider"],
        io_binding=lambda: binding,
        get_inputs=lambda: [
            SimpleNamespace(name="obs", type="tensor(float)", shape=[1, 2])
        ],
        get_outputs=lambda: [
            SimpleNamespace(
                name="actions", type="tensor(float)", shape=[1, 29]
            )
        ],
        run_with_iobinding=lambda value: value is binding,
    )
    synchronized, verified = [], []
    monkeypatch.setattr(torch.cuda, "synchronize", synchronized.append)
    monkeypatch.setattr(onnx_device, "verify_cuda_execution", verified.append)
    bound = TrainingBoundSession(session, "cuda:3")
    assert allocated == [torch.device("cuda:3"), torch.device("cuda:3")]
    assert bound.run() is bound.outputs
    assert synchronized == [torch.device("cuda:3")]
    assert verified == [session]
