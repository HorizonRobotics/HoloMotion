"""CUDA sessions with completed execution and explicit provider evidence."""

from pathlib import Path
import ctypes
import json
import tempfile

from .compiled_artifacts import available


class CudaDevice:
    type = "cuda"

    def __init__(self, index):
        self.index = index

    def __str__(self):
        return f"cuda:{self.index}"


def require_gpu(device="cuda:0"):
    if available():
        import warp as wp

        alias = str(device)
        if alias == "cuda":
            alias = "cuda:0"
        if not alias.startswith("cuda:"):
            raise RuntimeError(
                "A compatible CUDA GPU is required; CPU fallback is disabled"
            )
        selected = wp.get_device(alias)
        if not selected.is_cuda:
            raise RuntimeError(
                "A compatible CUDA GPU is required; CPU fallback is disabled"
            )
        return CudaDevice(selected.ordinal)

    import torch

    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "A compatible CUDA GPU is required; CPU fallback is disabled"
        )
    torch.cuda.get_device_properties(selected)
    return selected


def require_gpu_session(session):
    if (
        not session.get_providers()
        or session.get_providers()[0] != "CUDAExecutionProvider"
    ):
        raise RuntimeError(
            "CUDAExecutionProvider is required; CPU fallback is disabled"
        )


def require_cuda_driver(device_id):
    """Check GPU access before loading inference libraries, without Torch/Warp."""
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError as error:
        raise RuntimeError("A compatible CUDA driver is required") from error
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGetCount.restype = ctypes.c_int
    result = driver.cuInit(0)
    if result:
        raise RuntimeError(f"CUDA driver initialization failed: {result}")
    count = ctypes.c_int()
    result = driver.cuDeviceGetCount(ctypes.byref(count))
    if result or not 0 <= device_id < count.value:
        raise RuntimeError(f"CUDA device {device_id} is unavailable")


def create_device_session(
    path,
    provider="CUDAExecutionProvider",
    *,
    device_id=0,
    options=None,
    cuda_graph=False,
    cuda_options=None,
    runtime_stream=True,
):
    import onnxruntime as ort

    if provider != "CUDAExecutionProvider":
        raise RuntimeError("Only CUDA policy inference is supported")
    if runtime_stream:
        require_gpu(f"cuda:{device_id}")
    elif cuda_graph:
        raise ValueError("CUDA graphs require the runtime's fixed stream")
    else:
        require_cuda_driver(device_id)
    # The image supplies matching NVIDIA packages directly. Source-mode
    # PyTorch also uses these packages, but is not needed to expose them.
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "Install onnxruntime-gpu with a compatible CUDA runtime"
        )
    options = options or ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.use_deterministic_compute = True
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.enable_profiling = True
    scratch = tempfile.TemporaryDirectory(prefix="boxing-ort-")
    options.profile_file_prefix = str(Path(scratch.name) / "profile")
    provider_options = dict(cuda_options or {})
    provider_options["device_id"] = device_id
    provider_options["use_tf32"] = "0"
    if runtime_stream and available():
        import warp as wp

        provider_options.update(
            user_compute_stream=str(
                wp.get_stream(f"cuda:{device_id}").cuda_stream
            ),
            enable_cuda_graph="1" if cuda_graph else "0",
        )
    try:
        from .ordered_topk import execution_source

        session = ort.InferenceSession(
            execution_source(path),
            sess_options=options,
            providers=[("CUDAExecutionProvider", provider_options)],
        )
        session.disable_fallback()
        require_gpu_session(session)
        session.boxing_model_path = str(path)
        session.boxing_cuda_verified = False
        session.boxing_profile_directory = scratch
        return session
    except Exception:
        scratch.cleanup()
        raise


def verify_cuda_execution(session):
    require_gpu_session(session)
    if getattr(session, "boxing_cuda_verified", False):
        return
    profile = Path(session.end_profiling())
    try:
        records = json.loads(profile.read_text())
        providers = {
            item.get("args", {}).get("provider")
            for item in records
            if item.get("cat") == "Node"
            and item.get("args", {}).get("provider")
        }
        if providers != {"CUDAExecutionProvider"}:
            raise RuntimeError(
                f"Actual CUDA execution could not be verified: {sorted(providers)}"
            )
        session.boxing_cuda_verified = True
    finally:
        directory = getattr(session, "boxing_profile_directory", None)
        if directory is not None:
            directory.cleanup()
