"""Exact, stable TopK selection using bounded CUDA bitonic selections.

The pinned CUDA provider uses an unordered radix path for wide TopK inputs.
Select within contiguous blocks, then select among their candidates. Equal
values retain the lower original index, as required by ONNX. No epsilon,
weight change or expert approximation is introduced. Original files stay intact.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

BLOCK_WIDTH = 256
ORDERING_VERSION = "cuda_deterministic_no_tf32_stable_topk_v1"


def _graphs(graph):
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _graphs(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    yield from _graphs(child)


def _has_topk(model):
    return any(
        node.op_type == "TopK" and node.domain in ("", "ai.onnx")
        for root in [model.graph, *model.functions]
        for graph in _graphs(root)
        for node in graph.node
    )


def ordered_model(model):
    """Return a derived model for wide TopK, or None when no rewrite is needed.

    Wide selection requires a static last-axis width and constant K <= 128.
    Refuse unsupported cases rather than execute a known unordered kernel.
    Batch dimensions may remain dynamic. Nested graphs are inspected as well.
    """
    if not _has_topk(model):
        return None
    if model.functions:
        from onnx.inliner import inline_local_functions

        model = inline_local_functions(model)
    opset = next(
        (
            item.version
            for item in model.opset_import
            if item.domain in ("", "ai.onnx")
        ),
        0,
    )
    if opset < 11:
        raise ValueError("Ordered TopK requires ONNX opset >= 11")
    model = onnx.shape_inference.infer_shapes(model)
    changed = False
    counter = 0
    for graph in _graphs(model.graph):
        values = {
            item.name: item
            for item in [*graph.input, *graph.value_info, *graph.output]
        }
        constants = {
            item.name: numpy_helper.to_array(item)
            for item in graph.initializer
            if np.prod(item.dims) <= 16
        }
        for node in graph.node:
            if node.op_type == "Constant" and node.domain in ("", "ai.onnx"):
                for attribute in node.attribute:
                    if attribute.name == "value":
                        constants[node.output[0]] = numpy_helper.to_array(
                            attribute.t
                        )
        names = {
            name for node in graph.node for name in [*node.input, *node.output]
        }
        names.update(
            item.name
            for item in [*graph.input, *graph.output, *graph.initializer]
        )
        nodes = []
        for node in graph.node:
            if node.op_type != "TopK" or node.domain not in ("", "ai.onnx"):
                nodes.append(node)
                continue
            info = values.get(node.input[0])
            if info is None or not info.type.tensor_type.HasField("shape"):
                raise ValueError(
                    "Ordered TopK requires a known selection-axis width"
                )
            dimensions = info.type.tensor_type.shape.dim
            attrs = {
                item.name: helper.get_attribute_value(item)
                for item in node.attribute
            }
            axis = int(attrs.get("axis", -1))
            axis = axis + len(dimensions) if axis < 0 else axis
            width = dimensions[axis].dim_value
            if width <= 0:
                raise ValueError(
                    "Ordered TopK requires a static selection-axis width"
                )
            if width <= BLOCK_WIDTH:
                nodes.append(node)
                continue
            k_value = constants.get(node.input[1])
            if (
                axis != len(dimensions) - 1
                or k_value is None
                or np.size(k_value) != 1
            ):
                raise ValueError(
                    "Wide ordered TopK requires the last axis and constant K"
                )
            k = int(np.asarray(k_value).reshape(-1)[0])
            if not 0 < k <= BLOCK_WIDTH // 2:
                raise ValueError("Wide ordered TopK requires 0 < K <= 128")
            dtype = info.type.tensor_type.elem_type
            if dtype not in (
                onnx.TensorProto.FLOAT,
                onnx.TensorProto.FLOAT16,
                onnx.TensorProto.DOUBLE,
            ):
                raise ValueError("Wide ordered TopK requires a floating input")
            largest = int(attrs.get("largest", 1))
            while True:
                prefix = f"__boxing_ordered_topk_{counter}_"
                counter += 1
                if not any(name.startswith(prefix) for name in names):
                    break
            serial = 0

            def constant(value):
                nonlocal serial
                name = prefix + f"constant_{serial}"
                serial += 1
                graph.initializer.append(
                    numpy_helper.from_array(np.asarray(value), name)
                )
                return name

            def emit(op, inputs, **attributes):
                nonlocal serial
                name = prefix + f"value_{serial}"
                serial += 1
                nodes.append(
                    helper.make_node(op, inputs, [name], **attributes)
                )
                return name

            shape = emit("Shape", [node.input[0]])
            prefix_shape = emit(
                "Slice",
                [
                    shape,
                    constant(np.array([0], np.int64)),
                    constant(np.array([-1], np.int64)),
                ],
            )
            data, indices = node.input[0], None
            level = 0
            while width > BLOCK_WIDTH:
                count = (width + BLOCK_WIDTH - 1) // BLOCK_WIDTH
                if width % BLOCK_WIDTH:
                    padding = [0] * (2 * len(dimensions))
                    padding[-1] = count * BLOCK_WIDTH - width
                    pads = constant(np.array(padding, np.int64))
                    fill = constant(
                        np.array(
                            -np.inf if largest else np.inf,
                            dtype=helper.tensor_dtype_to_np_dtype(dtype),
                        )
                    )
                    data = emit("Pad", [data, pads, fill])
                    if indices is not None:
                        indices = emit(
                            "Pad",
                            [indices, pads, constant(np.array(0, np.int64))],
                        )
                chunk_shape = emit(
                    "Concat",
                    [
                        prefix_shape,
                        constant(np.array([count, BLOCK_WIDTH], np.int64)),
                    ],
                    axis=0,
                )
                chunk_data = emit("Reshape", [data, chunk_shape])
                selected, local_indices = (
                    prefix + f"selected_{level}",
                    prefix + f"indices_{level}",
                )
                nodes.append(
                    helper.make_node(
                        "TopK",
                        [chunk_data, node.input[1]],
                        [selected, local_indices],
                        axis=-1,
                        largest=largest,
                        sorted=1,
                    )
                )
                if indices is None:
                    offset_shape = [1] * (len(dimensions) - 1) + [count, 1]
                    offsets = constant(
                        (
                            np.arange(count, dtype=np.int64) * BLOCK_WIDTH
                        ).reshape(offset_shape)
                    )
                    selected_indices = emit("Add", [local_indices, offsets])
                else:
                    chunk_indices = emit("Reshape", [indices, chunk_shape])
                    selected_indices = emit(
                        "GatherElements",
                        [chunk_indices, local_indices],
                        axis=-1,
                    )
                width = count * k
                flat_shape = emit(
                    "Concat",
                    [prefix_shape, constant(np.array([width], np.int64))],
                    axis=0,
                )
                data = emit("Reshape", [selected, flat_shape])
                indices = emit("Reshape", [selected_indices, flat_shape])
                level += 1
            final_indices = prefix + "final_local_indices"
            nodes.append(
                helper.make_node(
                    "TopK",
                    [data, node.input[1]],
                    [node.output[0], final_indices],
                    axis=-1,
                    largest=largest,
                    sorted=1,
                )
            )
            nodes.append(
                helper.make_node(
                    "GatherElements",
                    [indices, final_indices],
                    [node.output[1]],
                    axis=-1,
                )
            )
            changed = True
        del graph.node[:]
        graph.node.extend(nodes)
    return model if changed else None


@lru_cache(maxsize=4)
def _serialized(path, size, modified):
    derived = ordered_model(onnx.load(path))
    return derived.SerializeToString() if derived is not None else None


def execution_source(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    serialized = _serialized(str(path), stat.st_size, stat.st_mtime_ns)
    return serialized if serialized is not None else str(path)
