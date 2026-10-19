"""Install the one public HoloMotion V2 GPU evaluation runtime."""

from .onnx_device import require_gpu

_runtime = None


def install_runtime(lower_directory):
    global _runtime
    require_gpu()
    from . import lower

    lower.configure_bundle(lower_directory)
    if _runtime is None:
        from . import match_runtime
        from .compiled_artifacts import available

        if available():
            from .lean_runtime import install
        else:
            from .fast_runtime import install

        install()
        match_runtime.HoloMotionV2Policy = lower.LinkPolicy
        _runtime = match_runtime
    return _runtime


def verify_result_metadata(metadata):
    if metadata.get("holomotion_version") != "2.0_link_10500":
        raise ValueError("The frozen HoloMotion V2 controller is required")
    if metadata.get(
        "physics_execution"
    ) != "gpu_resident_cuda_graph" or not str(
        metadata.get("physics_device", "")
    ).startswith("cuda"):
        raise ValueError("Official physics must execute on the GPU")
    if (
        metadata.get("upper_policy_provider") != "CUDAExecutionProvider"
        or metadata.get("lower_policy_provider")
        != ["CUDAExecutionProvider"] * 2
    ):
        raise ValueError(
            "Both upper and lower policies require CUDA inference"
        )
    if metadata.get("upper_cuda_verified") != [True, True] or metadata.get(
        "lower_cuda_verified"
    ) != [True, True]:
        raise ValueError("Actual CUDA execution was not verified")
    if metadata.get(
        "upper_policy_contract"
    ) != "holomotion_v2_boxing_obs1097_target_pose225_v1" or metadata.get(
        "lower_reference_frame_offsets"
    ) != [1, 2, 3, 4, 5]:
        raise ValueError(
            "The public interface or future reference frames differ"
        )
