"""ONNX Runtime helpers for the 29DOF policy node."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Any

import numpy as np


TENSORRT_PROVIDER = "TensorrtExecutionProvider"
TENSORRT_REQUIRED_SHARED_LIBS = ("libnvinfer.so.8",)
DEFAULT_TENSORRT_DEVICE_ID = 0
DEFAULT_TENSORRT_ENGINE_CACHE_PATH = "./.cache/tensorrt_engines"


def _provider_bool(value: bool) -> str:
    return "True" if bool(value) else "False"


@dataclass(frozen=True)
class MotionPolicyIo:
    input_name: str
    output_name: str
    kv_input_name: str | None
    kv_output_name: str | None
    kv_shape: Any
    step_idx_input_name: str | None
    kv_dtype: type[np.generic]


@dataclass(frozen=True)
class MotionKvCache:
    enabled: bool
    cache: np.ndarray | None
    shape: list[int]
    model_context_len: int
    effective_context_len: int


@dataclass(frozen=True)
class DualPolicyBundle:
    velocity_session: Any
    motion_session: Any
    velocity_onnx_path: str
    motion_onnx_path: str
    velocity_input_name: str
    velocity_output_name: str
    motion_io: MotionPolicyIo
    motion_kv_cache: MotionKvCache
    inference_backend: str


def normalize_inference_backend(value: str) -> str:
    backend = str(value or "onnx").strip().lower()
    aliases = {
        "onnxruntime": "onnx",
        "ort": "onnx",
        "cuda": "onnx",
        "trt": "tensorrt",
        "tensor_rt": "tensorrt",
    }
    backend = aliases.get(backend, backend)
    if backend not in {"onnx", "tensorrt"}:
        raise ValueError("inference_backend must be 'onnx' or 'tensorrt'")
    return backend


def build_onnxruntime_providers(device_id: int = 0) -> list[Any]:
    return [
        (
            "CUDAExecutionProvider",
            {
                "device_id": int(device_id),
            },
        ),
        "CPUExecutionProvider",
    ]


def build_tensorrt_providers(
    device_id: int = DEFAULT_TENSORRT_DEVICE_ID,
    fp16_enable: bool = False,
    engine_cache_enable: bool = True,
    engine_cache_path: str = DEFAULT_TENSORRT_ENGINE_CACHE_PATH,
    available_providers: list[str] | None = None,
) -> list[Any]:
    should_check_shared_libraries = available_providers is None
    if available_providers is None:
        import onnxruntime

        available_providers = list(onnxruntime.get_available_providers())

    if TENSORRT_PROVIDER not in available_providers:
        raise RuntimeError(
            "inference_backend='tensorrt' was requested, but ONNX Runtime "
            f"available providers are {available_providers}. Install/use an "
            "onnxruntime-gpu build with TensorRTExecutionProvider."
        )
    if should_check_shared_libraries:
        check_tensorrt_shared_libraries()

    trt_options: dict[str, Any] = {
        "device_id": str(int(device_id)),
        "trt_fp16_enable": _provider_bool(fp16_enable),
        "trt_engine_cache_enable": _provider_bool(engine_cache_enable),
    }
    cache_path = str(engine_cache_path or "").strip()
    if bool(engine_cache_enable) and cache_path:
        os.makedirs(cache_path, exist_ok=True)
        trt_options["trt_engine_cache_path"] = cache_path

    return [(TENSORRT_PROVIDER, trt_options)]


def check_tensorrt_shared_libraries(
    library_names: tuple[str, ...] = TENSORRT_REQUIRED_SHARED_LIBS,
) -> None:
    missing: list[str] = []
    loader_errors: list[str] = []
    for library_name in library_names:
        try:
            ctypes.CDLL(library_name)
        except OSError as exc:
            missing.append(library_name)
            loader_errors.append(f"{library_name}: {exc}")
    if missing:
        raise RuntimeError(
            "inference_backend='tensorrt' requires TensorRT runtime libraries, "
            f"but these shared libraries are not loadable: {', '.join(missing)}. "
            "Install the matching TensorRT runtime in the container or add it to "
            "LD_LIBRARY_PATH. Loader errors: "
            + " | ".join(loader_errors)
        )


def build_inference_providers(
    inference_backend: str = "onnx",
    device_id: int = DEFAULT_TENSORRT_DEVICE_ID,
    tensorrt_fp16_enable: bool = False,
    tensorrt_engine_cache_enable: bool = True,
    tensorrt_engine_cache_path: str = DEFAULT_TENSORRT_ENGINE_CACHE_PATH,
    available_providers: list[str] | None = None,
) -> list[Any]:
    backend = normalize_inference_backend(inference_backend)
    if backend == "onnx":
        return build_onnxruntime_providers(device_id=device_id)
    return build_tensorrt_providers(
        device_id=device_id,
        fp16_enable=tensorrt_fp16_enable,
        engine_cache_enable=tensorrt_engine_cache_enable,
        engine_cache_path=tensorrt_engine_cache_path,
        available_providers=available_providers,
    )


def create_onnx_session_options(
    intra_op_threads: int,
    disable_cpu_ep_fallback: bool = False,
):
    import onnxruntime

    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = int(intra_op_threads)
    sess_options.inter_op_num_threads = 1
    if bool(disable_cpu_ep_fallback):
        sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return sess_options


def create_onnx_session(
    onnx_path: str,
    sess_options,
    providers: list[Any],
    disable_session_fallback: bool = False,
):
    import onnxruntime

    session = onnxruntime.InferenceSession(
        str(onnx_path), sess_options=sess_options, providers=providers
    )
    if bool(disable_session_fallback) and hasattr(session, "disable_fallback"):
        session.disable_fallback()
    return session


def assert_required_provider(session, provider_name: str, model_path: str) -> None:
    actual_providers = list(session.get_providers())
    if provider_name not in actual_providers:
        raise RuntimeError(
            f"Policy session for {model_path} did not enable required provider "
            f"{provider_name}. Actual providers: {actual_providers}"
        )


def find_first_onnx_file(exported_model_dir: str) -> str:
    onnx_files = [
        filename
        for filename in os.listdir(exported_model_dir)
        if filename.endswith(".onnx")
    ]
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX files found in {exported_model_dir}")
    return os.path.join(exported_model_dir, onnx_files[0])


def resolve_policy_onnx_path(package_share_dir: str, model_folder: str) -> str:
    exported_model_dir = os.path.join(
        package_share_dir,
        "models",
        model_folder,
        "exported",
    )
    return find_first_onnx_file(exported_model_dir)


def load_dual_policy_bundle(
    package_share_dir: str,
    velocity_model_folder: str,
    motion_model_folder: str,
    intra_op_threads: int,
    motion_max_context_len: int,
    inference_backend: str = "onnx",
    device_id: int = DEFAULT_TENSORRT_DEVICE_ID,
    tensorrt_fp16_enable: bool = False,
    tensorrt_engine_cache_enable: bool = True,
    tensorrt_engine_cache_path: str = DEFAULT_TENSORRT_ENGINE_CACHE_PATH,
    providers: list[Any] | None = None,
) -> DualPolicyBundle:
    backend = normalize_inference_backend(inference_backend)
    providers = (
        build_inference_providers(
            inference_backend=backend,
            device_id=device_id,
            tensorrt_fp16_enable=tensorrt_fp16_enable,
            tensorrt_engine_cache_enable=tensorrt_engine_cache_enable,
            tensorrt_engine_cache_path=tensorrt_engine_cache_path,
        )
        if providers is None
        else list(providers)
    )
    strict_tensorrt = backend == "tensorrt"
    sess_options = create_onnx_session_options(
        intra_op_threads,
        disable_cpu_ep_fallback=strict_tensorrt,
    )

    velocity_onnx_path = resolve_policy_onnx_path(
        package_share_dir, velocity_model_folder
    )
    motion_onnx_path = resolve_policy_onnx_path(
        package_share_dir, motion_model_folder
    )

    velocity_session = create_onnx_session(
        velocity_onnx_path,
        sess_options=sess_options,
        providers=providers,
        disable_session_fallback=strict_tensorrt,
    )
    motion_session = create_onnx_session(
        motion_onnx_path,
        sess_options=sess_options,
        providers=providers,
        disable_session_fallback=strict_tensorrt,
    )
    if strict_tensorrt:
        assert_required_provider(
            velocity_session,
            TENSORRT_PROVIDER,
            velocity_onnx_path,
        )
        assert_required_provider(
            motion_session,
            TENSORRT_PROVIDER,
            motion_onnx_path,
        )
    velocity_input_name, velocity_output_name = get_primary_io_names(
        velocity_session
    )
    motion_input_name, _ = get_primary_io_names(motion_session)
    motion_io = inspect_motion_policy_io(
        motion_session.get_inputs(),
        motion_session.get_outputs(),
        default_input_name=motion_input_name,
    )
    motion_kv_cache = create_motion_kv_cache(
        kv_input_name=motion_io.kv_input_name,
        kv_shape=motion_io.kv_shape,
        kv_dtype=motion_io.kv_dtype,
        max_context_len=motion_max_context_len,
    )

    return DualPolicyBundle(
        velocity_session=velocity_session,
        motion_session=motion_session,
        velocity_onnx_path=velocity_onnx_path,
        motion_onnx_path=motion_onnx_path,
        velocity_input_name=velocity_input_name,
        velocity_output_name=velocity_output_name,
        motion_io=motion_io,
        motion_kv_cache=motion_kv_cache,
        inference_backend=backend,
    )


def get_primary_io_names(session) -> tuple[str, str]:
    return session.get_inputs()[0].name, session.get_outputs()[0].name


def infer_onnx_dim(dim, default: int = 1) -> int:
    if isinstance(dim, int) and dim > 0:
        return dim
    return int(default)


def infer_numpy_dtype_from_onnx_type(type_str: str):
    type_str = str(type_str).lower()
    if "float16" in type_str:
        return np.float16
    if "float64" in type_str or "double" in type_str:
        return np.float64
    if "int64" in type_str:
        return np.int64
    if "int32" in type_str:
        return np.int32
    if "bool" in type_str:
        return np.bool_
    return np.float32


def build_dummy_input_from_onnx_node(
    node, fallback_last_dim: int | None = None
) -> np.ndarray:
    shape = list(getattr(node, "shape", []) or [])
    if not shape:
        shape = [1]
    inferred_shape = [infer_onnx_dim(dim, default=1) for dim in shape]
    if fallback_last_dim is not None and len(inferred_shape) >= 2:
        last_dim = shape[-1]
        if not isinstance(last_dim, int) or last_dim <= 0:
            inferred_shape[-1] = int(fallback_last_dim)
    dtype = infer_numpy_dtype_from_onnx_type(
        getattr(node, "type", "tensor(float)")
    )
    return np.zeros(inferred_shape, dtype=dtype)


def inspect_motion_policy_io(
    input_nodes,
    output_nodes,
    default_input_name: str,
) -> MotionPolicyIo:
    motion_input_name = default_input_name
    motion_kv_input_name = None
    motion_kv_shape = None
    motion_step_idx_input_name = None
    motion_kv_dtype = np.float32

    for node in input_nodes:
        name = node.name
        node_type = node.type
        if "obs" in name:
            motion_input_name = name
        elif "past_key_values" in name:
            motion_kv_input_name = name
            motion_kv_shape = node.shape
            if isinstance(node_type, str) and "float16" in node_type:
                motion_kv_dtype = np.float16
        elif "step_idx" in name or name == "step_idx":
            motion_step_idx_input_name = name
        elif "current_pos" in name or name == "current_pos":
            motion_step_idx_input_name = name
        elif (
            motion_step_idx_input_name is None
            and isinstance(node_type, str)
            and "int64" in node_type
            and name not in (motion_input_name, motion_kv_input_name)
        ):
            motion_step_idx_input_name = name

    action_output_name = None
    kv_output_name = None
    for node in output_nodes:
        if "present_key_values" in node.name:
            kv_output_name = node.name
        elif "actions" in node.name:
            action_output_name = node.name
    if action_output_name is None:
        for node in output_nodes:
            if kv_output_name is not None and node.name == kv_output_name:
                continue
            action_output_name = node.name
            break
    if action_output_name is None:
        action_output_name = output_nodes[0].name

    return MotionPolicyIo(
        input_name=motion_input_name,
        output_name=action_output_name,
        kv_input_name=motion_kv_input_name,
        kv_output_name=kv_output_name,
        kv_shape=motion_kv_shape,
        step_idx_input_name=motion_step_idx_input_name,
        kv_dtype=motion_kv_dtype,
    )


def create_motion_kv_cache(
    kv_input_name: str | None,
    kv_shape,
    kv_dtype,
    max_context_len: int,
) -> MotionKvCache:
    if kv_input_name and kv_shape:
        shape = [dim if isinstance(dim, int) else 1 for dim in kv_shape]
        cache = np.zeros(shape, dtype=kv_dtype)
        model_context_len = int(shape[3]) if len(shape) > 3 else 0
        max_context_len = int(max_context_len)
        if max_context_len > 0 and model_context_len > 0:
            effective_context_len = min(max_context_len, model_context_len)
        else:
            effective_context_len = model_context_len
        return MotionKvCache(
            enabled=True,
            cache=cache,
            shape=shape,
            model_context_len=model_context_len,
            effective_context_len=effective_context_len,
        )

    return MotionKvCache(
        enabled=False,
        cache=None,
        shape=[],
        model_context_len=0,
        effective_context_len=0,
    )


def warmup_motion_policy(
    session,
    input_name: str,
    output_name: str,
    kv_input_name: str | None,
    kv_output_name: str | None,
    step_idx_input_name: str | None,
    use_kv_cache: bool,
    kv_cache: np.ndarray | None,
    kv_shape,
    kv_dtype,
    obs_dim: int | None,
    num_iters: int = 2,
) -> int:
    if session is None:
        return 0

    input_nodes = {node.name: node for node in session.get_inputs()}
    obs_node = input_nodes.get(input_name, None)
    if obs_node is None:
        raise ValueError(
            f"Motion warmup failed to find obs input '{input_name}'."
        )

    obs_dummy = build_dummy_input_from_onnx_node(
        obs_node,
        fallback_last_dim=obs_dim,
    )
    output_names = [output_name]
    if kv_output_name:
        output_names.append(kv_output_name)

    local_kv_cache = None
    if use_kv_cache and kv_input_name is not None:
        if kv_cache is not None:
            local_kv_cache = np.zeros_like(kv_cache)
        else:
            shape = [infer_onnx_dim(dim, default=1) for dim in (kv_shape or [])]
            local_kv_cache = np.zeros(shape, dtype=kv_dtype)

    iterations = max(1, int(num_iters))
    for warmup_step in range(iterations):
        input_feed = {input_name: obs_dummy}
        if use_kv_cache and kv_input_name is not None:
            input_feed[kv_input_name] = local_kv_cache
        if step_idx_input_name is not None:
            step_node = input_nodes.get(step_idx_input_name, None)
            step_dtype = np.int64
            if step_node is not None:
                step_dtype = infer_numpy_dtype_from_onnx_type(
                    getattr(step_node, "type", "tensor(int64)")
                )
            input_feed[step_idx_input_name] = np.array(
                [warmup_step], dtype=step_dtype
            )

        warmup_output = session.run(output_names, input_feed)
        if local_kv_cache is not None and kv_output_name and len(warmup_output) > 1:
            local_kv_cache = warmup_output[1]

    return iterations


def warmup_policy_session(
    session,
    input_name: str,
    output_name: str,
    obs_dim: int | None,
    num_iters: int = 2,
) -> int:
    if session is None:
        return 0

    input_nodes = {node.name: node for node in session.get_inputs()}
    obs_node = input_nodes.get(input_name, None)
    if obs_node is None:
        raise ValueError(f"Warmup failed to find obs input '{input_name}'.")

    obs_dummy = build_dummy_input_from_onnx_node(
        obs_node,
        fallback_last_dim=obs_dim,
    )
    iterations = max(1, int(num_iters))
    for _ in range(iterations):
        session.run([output_name], {input_name: obs_dummy})
    return iterations


def read_onnx_metadata(onnx_model_path: str) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(onnx_model_path))
    meta = {prop.key: prop.value for prop in model.metadata_props}

    def _parse_floats(csv_str: str) -> np.ndarray:
        return np.array(
            [float(value) for value in csv_str.split(",") if value != ""],
            dtype=np.float32,
        )

    def _parse_optional_int(value: str | None) -> int | None:
        if value is None or str(value).strip() == "":
            return None
        return int(str(value).strip())

    rope_max_seq_len = _parse_optional_int(meta.get("rope_max_seq_len"))
    parsed = {
        "action_scale": _parse_floats(meta["action_scale"]),
        "kps": _parse_floats(meta["joint_stiffness"]),
        "kds": _parse_floats(meta["joint_damping"]),
        "default_joint_pos": _parse_floats(meta["default_joint_pos"]),
        "joint_names": [
            value for value in meta["joint_names"].split(",") if value != ""
        ],
    }
    if rope_max_seq_len is not None:
        parsed["rope_max_seq_len"] = int(rope_max_seq_len)
    return parsed
