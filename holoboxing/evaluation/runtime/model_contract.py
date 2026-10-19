"""The participant ONNX contract, independent of a competition service."""

from pathlib import Path
import hashlib

CONTRACT = "holomotion_v2_boxing_obs1097_target_pose225_v1"
MAX_MODEL_BYTES = 50_000_000
INPUT_LINK_NAMES = (
    "pelvis",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check_model(path):
    import onnx

    path = Path(path).resolve(strict=True)
    size = path.stat().st_size
    if size >= MAX_MODEL_BYTES:
        raise ValueError(
            f"Model size {size} bytes must be < {MAX_MODEL_BYTES} bytes"
        )
    model = onnx.load(path, load_external_data=False)
    if any(
        tensor.data_location == onnx.TensorProto.EXTERNAL
        or tensor.external_data
        for tensor in _tensors(model)
    ):
        raise ValueError(
            "Submit a self-contained ONNX file; external tensor files are unsupported"
        )
    onnx.checker.check_model(model, full_check=True)
    return model, size


def _tensors(message):
    """Include sparse tensors, nested graphs and local-function attributes."""
    import onnx

    if isinstance(message, onnx.TensorProto):
        yield message
    for descriptor, value in message.ListFields():
        if descriptor.type == descriptor.TYPE_MESSAGE:
            for child in value if descriptor.is_repeated else (value,):
                yield from _tensors(child)


def check_contract(graph):
    import onnx

    for nodes, name, width in (
        (graph.graph.input, "obs", 1097),
        (graph.graph.output, "target_pose", 225),
    ):
        if len(nodes) != 1 or nodes[0].name != name:
            raise ValueError(f"Expected one float32 {name}[B,{width}]")
        tensor = nodes[0].type.tensor_type
        if (
            tensor.elem_type != onnx.TensorProto.FLOAT
            or len(tensor.shape.dim) != 2
        ):
            raise ValueError(f"Expected float32 {name}[B,{width}]")
        batch, columns = tensor.shape.dim
        if batch.HasField("dim_value") and batch.dim_value != 1:
            raise ValueError("The policy must support batch=1")
        if not columns.HasField("dim_value") or columns.dim_value != width:
            raise ValueError(f"Expected {name}[B,{width}]")
    metadata = {item.key: item.value for item in graph.metadata_props}
    if metadata.get("policy_contract", CONTRACT) != CONTRACT:
        raise ValueError(f"Expected policy_contract={CONTRACT}")
    _check_deterministic_graphs(graph)
    from .ordered_topk import ordered_model

    ordered_model(graph)


def _check_deterministic_graphs(model):
    """Inspect nested graphs and local functions, including branches not probed."""
    import onnx
    from onnx import numpy_helper

    random_ops = {
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniform",
        "RandomUniformLike",
        "Multinomial",
        "Bernoulli",
    }
    standard_opset = next(
        (
            x.version
            for x in model.opset_import
            if not x.domain or x.domain == "ai.onnx"
        ),
        0,
    )
    for container in _messages(model, (onnx.GraphProto, onnx.FunctionProto)):
        false_constants = set()
        for value in getattr(container, "initializer", ()):
            if value.data_type == onnx.TensorProto.BOOL:
                array = numpy_helper.to_array(value)
                if array.size == 1 and not bool(array.reshape(-1)[0]):
                    false_constants.add(value.name)
        for node in container.node:
            if node.op_type == "Constant" and node.domain in ("", "ai.onnx"):
                for attribute in node.attribute:
                    if (
                        attribute.name == "value"
                        and attribute.HasField("t")
                        and attribute.t.data_type == onnx.TensorProto.BOOL
                    ):
                        value = numpy_helper.to_array(attribute.t)
                        if value.size == 1 and not bool(value.reshape(-1)[0]):
                            false_constants.update(node.output)
        opset = next(
            (
                x.version
                for x in getattr(container, "opset_import", ())
                if x.domain in ("", "ai.onnx")
            ),
            standard_opset,
        )
        for node in container.node:
            if node.domain not in ("", "ai.onnx"):
                continue
            if node.op_type in random_ops:
                raise ValueError(
                    f"Reproducible policies cannot use random operator {node.op_type}"
                )
            if node.op_type == "Dropout" and (
                opset < 12
                or (
                    len(node.input) >= 3
                    and node.input[2]
                    and node.input[2] not in false_constants
                )
            ):
                raise ValueError(
                    "Dropout training_mode must be absent or a constant false in its graph"
                )


def _messages(message, kinds):
    if isinstance(message, kinds):
        yield message
    for descriptor, value in message.ListFields():
        if descriptor.type == descriptor.TYPE_MESSAGE:
            for child in value if descriptor.is_repeated else (value,):
                yield from _messages(child, kinds)
