"""Participant compatibility and official full-chain qualification reports."""

from datetime import datetime, timezone
from pathlib import Path

from .qualification import qualification
from .runtime.model_contract import (
    CONTRACT,
    MAX_MODEL_BYTES,
    check_model,
    check_contract,
)
from .runtime.policy_output import CompetitorOutputError


def validate_model(
    model,
    assets,
    lower,
    *,
    qualification_mode="local",
    max_model_bytes=MAX_MODEL_BYTES,
):
    if qualification_mode not in ("local", "official"):
        raise ValueError("Qualification must be local or official")
    if (
        type(max_model_bytes) is not int
        or not 1 <= max_model_bytes <= MAX_MODEL_BYTES
    ):
        raise ValueError(
            "The model size limit must be a positive integer no greater than 50,000,000 bytes"
        )
    report = dict(
        date=datetime.now(timezone.utc).isoformat(),
        status="FAIL",
        actual_fps=0.0,
        qualification_mode=qualification_mode,
        status_scope="local interface and runtime compatibility"
        if qualification_mode == "local"
        else "official hardware and complete-chain performance",
        format_check=dict(
            expected="valid self-contained ONNX", status="NOT_RUN"
        ),
        size_check=dict(
            expected=f"strictly <{max_model_bytes:,} bytes",
            status="NOT_RUN",
            total_bytes=0,
            limit_bytes_exclusive=max_model_bytes,
        ),
        contract_check=dict(expected=CONTRACT, status="NOT_RUN"),
        execution_provider="CUDAExecutionProvider",
        error_code="",
        error_reason="",
        lower_controller=dict(
            version="2.0_link_10500",
            command_mode="VR-5",
            model_path=str(Path(lower) / "exported/model_10500.onnx"),
        ),
    )
    stage = "size"
    try:
        model = Path(model)
        size = model.stat().st_size
        report["size_check"].update(
            total_bytes=size,
            status="PASS" if size < max_model_bytes else "FAIL",
        )
        if size >= max_model_bytes:
            raise ValueError(
                f"Model size {size} bytes must be < {max_model_bytes} bytes"
            )
        stage = "format"
        graph, _ = check_model(model)
        report["format_check"]["status"] = "PASS"
        stage = "schema"
        check_contract(graph)
        stage = "provider"
        from .runtime.onnx_device import create_device_session
        from .runtime.upper import neutral_observation, schema_for_model
        from .runtime.policy_output import run_competitor, sanitize_target_pose

        session = create_device_session(model)
        stage = "probe"
        schema = schema_for_model(graph)
        _, sanitation = sanitize_target_pose(
            run_competitor(session, neutral_observation(schema))
        )
        report["contract_check"].update(
            status="PASS",
            input_shape=[1, 1097],
            output_shape=[1, 225],
            history_frames=10,
            frame_offsets=[1, 2, 3, 4, 5],
            source_hashes=schema.source_hashes,
            output_sanitization=sanitation,
        )
        # The temporary interface probe is not part of timed samples.
        del session, graph
        stage = "chain"
        from .runtime.chain_benchmark import benchmark_model

        chain = benchmark_model(
            model,
            assets=assets,
            lower_directory=lower,
            warmup=100,
            iterations=1000,
        )
        report["chain_benchmark"] = chain
        report["actual_fps"] = chain["fps"] or 0.0
        hardware = qualification(chain["gpu"], chain["mean_ms"])
        chain.update(
            required_gpu=hardware["required_gpu"],
            target_hardware_match=hardware["target_hardware_match"],
        )
        report["hardware_qualification"] = hardware
        if (
            qualification_mode == "official"
            and not hardware["target_hardware_match"]
        ):
            report.update(
                error_code="PLATFORM_WRONG_TARGET_GPU",
                error_reason=f"Required {hardware['required_gpu']}; measured {chain['gpu']}. {hardware['scope_notice']}",
            )
        elif qualification_mode == "official" and not chain["passed"]:
            report.update(
                error_code="CHAIN_MEAN_LATENCY_EXCEEDED",
                error_reason=f"Upper + HoloMotion mean {chain['mean_ms']:.6f} ms exceeds 5 ms after warmup",
            )
        else:
            report["status"] = "PASS"
    except Exception as error:
        model_fault = isinstance(error, CompetitorOutputError) or (
            stage in ("size", "format", "schema")
            and not isinstance(error, (OSError, MemoryError))
        )
        if model_fault:
            code = (
                "MODEL_TOO_LARGE"
                if stage == "size"
                else "FORMAT_CHECK_FAILED"
                if stage == "format"
                else "CONTRACT_CHECK_FAILED"
            )
            report.update(
                error_code=code,
                error_reason=f"{type(error).__name__}: {error}",
            )
            if stage == "format":
                report["format_check"]["status"] = "FAIL"
            if stage == "schema" or isinstance(error, CompetitorOutputError):
                report["contract_check"]["status"] = "FAIL"
        else:
            report.update(
                error_code="PLATFORM_CHAIN_VALIDATION_FAILED",
                error_reason=f"{type(error).__name__}: {error}",
            )
    return report
