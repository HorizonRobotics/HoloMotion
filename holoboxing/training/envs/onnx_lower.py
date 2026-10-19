"""Batch specialization of the pinned public ONNX, without changing weights."""

from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from holoboxing.evaluation.models import frozen_manifest
from holoboxing.evaluation.runtime.fast_runtime import BoundSession
from holoboxing.evaluation.runtime.model_contract import sha256
from holoboxing.evaluation.runtime.onnx_device import create_device_session


def specialize_graph(path, batch_size):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    manifest = frozen_manifest()
    expected_model = manifest["files"][manifest["model"]]
    if sha256(path) != expected_model["sha256"]:
        raise ValueError(
            "Only the pinned public lower ONNX can be specialized"
        )
    graph = onnx.load(path)
    # These constants are the batch-bearing reshapes in this exact release.
    shapes = {
        10: (1, 1, 8, 64),
        11: (1, 1, 2, 4, 64),
        30: (1, 1, 1, 1),
        31: (1, 1, 4, 64),
        40: (1, 4, 2, 32, 64),
        43: (1, 8, 32, 64),
        45: (1, 4, 2, 32, 64),
        48: (1, 8, 32, 64),
        55: (1, 1, -1),
        58: (1, 512),
        59: (1, 2),
        60: (1, 2),
        62: (1, 2, 512),
        63: (1, 2, 512),
        66: (2, 512),
        67: (2,),
        79: (1, 1, 512),
    }
    remaining = set(shapes)
    for node in graph.graph.node:
        for index, expected in shapes.items():
            if node.name != f"/actor_module/Constant_{index}":
                continue
            actual = numpy_helper.to_array(node.attribute[0].t)
            if tuple(actual) != expected:
                raise ValueError("Frozen V2 specialization constant changed")
            updated = np.asarray(expected, dtype=np.int64)
            updated[0] *= batch_size
            node.attribute[0].t.CopyFrom(numpy_helper.from_array(updated))
            remaining.remove(index)
    if remaining:
        raise ValueError("Frozen V2 specialization nodes are missing")
    for value in (*graph.graph.input, *graph.graph.output):
        axis = (
            2 if value.name in ("past_key_values", "present_key_values") else 0
        )
        value.type.tensor_type.shape.dim[axis].dim_value = batch_size
    del graph.graph.value_info[:]
    onnx.checker.check_model(graph)
    return graph


class OnnxLower:
    """Shared inference; recurrent state belongs to each controller."""

    def __init__(
        self, path, batch_size, device, directory, inference_batch_size=128
    ):
        if any(
            type(value) is not int or value < 1
            for value in (batch_size, inference_batch_size)
        ):
            raise ValueError("Batch sizes must be positive integers")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("The frozen ONNX controller requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.batch_size = batch_size
        # Keep static ONNX batches without restricting the environment count.
        self.chunk_size = next(
            size for size in range(min(batch_size, inference_batch_size), 0, -1)
            if batch_size % size == 0
        )
        graph = specialize_graph(path, self.chunk_size)
        self.metadata = {item.key: item.value for item in graph.metadata_props}
        target = Path(directory) / f"lower_batch_{self.chunk_size}.onnx"
        target.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(graph, target)
        del graph
        self.session = create_device_session(
            target,
            device_id=self.device.index,
            cuda_options={
                "arena_extend_strategy": "kSameAsRequested",
                "user_compute_stream": str(
                    torch.cuda.current_stream(self.device).cuda_stream
                ),
            },
        )
        expected = {
            "obs": [self.chunk_size, 1378],
            "past_key_values": [1, 2, self.chunk_size, 32, 4, 64],
            "step_idx": [self.chunk_size],
        }
        if {n.name: n.shape for n in self.session.get_inputs()} != expected:
            raise ValueError("Public ONNX input contract differs")
        outputs = {n.name: n.shape for n in self.session.get_outputs()}
        if (
            outputs.get("actions") != [self.chunk_size, 29]
            or outputs.get("present_key_values") != expected["past_key_values"]
        ):
            raise ValueError("Public ONNX output contract differs")
        self.bound = BoundSession(self.session, self.device)

    @torch.inference_mode()
    def __call__(self, observation, cache, step):
        actions = torch.empty((self.batch_size, 29), device=self.device)
        present = torch.empty_like(cache)
        for start in range(0, self.batch_size, self.chunk_size):
            rows = slice(start, start + self.chunk_size)
            self.bound.inputs["obs"].copy_(observation[rows])
            self.bound.inputs["past_key_values"].copy_(cache[:, :, rows])
            self.bound.inputs["step_idx"].copy_(step[rows])
            # ORT may use its own stream (CUDA stream 0 is not an external
            # stream). Finish input writes and the previous output reads before
            # reusing the bound buffers; BoundSession fences ORT's output writes.
            torch.cuda.current_stream(self.device).synchronize()
            output = self.bound.run()
            actions[rows].copy_(output["actions"])
            present[:, :, rows].copy_(output["present_key_values"])
        return actions, present
