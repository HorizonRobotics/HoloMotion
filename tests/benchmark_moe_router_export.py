import re
import time
from pathlib import Path

import torch


def _extract_int_setting(config_path: Path, key: str) -> int:
    pattern = re.compile(rf"^\s*{re.escape(key)}:\s*([0-9]+)\s*$")
    for line in config_path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            return int(match.group(1))
    raise ValueError(
        f"Could not find integer setting {key!r} in {config_path}"
    )


def _load_b0310_shape_config() -> dict[str, int]:
    repo_root = Path(__file__).resolve().parents[1]
    module_cfg = (
        repo_root
        / "holomotion"
        / "config"
        / "modules"
        / "motion_tracking"
        / "tf_motrack_v3.yaml"
    )
    return {
        "num_fine_experts": _extract_int_setting(
            module_cfg, "num_fine_experts"
        ),
        "top_k": _extract_int_setting(module_cfg, "top_k"),
        "max_ctx_len": _extract_int_setting(module_cfg, "max_ctx_len"),
    }


def _router_scores_training(
    logits_fp32: torch.Tensor,
    *,
    top_k: int,
    bias_fp32: torch.Tensor | None = None,
) -> torch.Tensor:
    choice_logits = (
        logits_fp32 if bias_fp32 is None else logits_fp32 + bias_fp32
    )
    _, topk_idx = torch.topk(choice_logits, top_k, dim=-1)
    selected_logits = logits_fp32.gather(-1, topk_idx)
    log_z = torch.logsumexp(logits_fp32, dim=-1, keepdim=True)
    selected_probs = torch.exp(selected_logits - log_z)
    return selected_probs / selected_probs.sum(dim=-1, keepdim=True).clamp_min(
        1.0e-20
    )


def _router_scores_export_safe(
    logits_fp32: torch.Tensor,
    *,
    top_k: int,
    bias_fp32: torch.Tensor | None = None,
) -> torch.Tensor:
    choice_logits = (
        logits_fp32 if bias_fp32 is None else logits_fp32 + bias_fp32
    )
    _, topk_idx = torch.topk(choice_logits, top_k, dim=-1)
    selected_probs = torch.softmax(logits_fp32, dim=-1).gather(-1, topk_idx)
    return selected_probs / selected_probs.sum(dim=-1, keepdim=True).clamp_min(
        1.0e-20
    )


def _benchmark(
    fn,
    logits_fp32: torch.Tensor,
    *,
    top_k: int,
    bias_fp32: torch.Tensor | None = None,
    warmup_iters: int = 200,
    measure_iters: int = 2000,
) -> float:
    is_cuda = logits_fp32.is_cuda
    with torch.inference_mode():
        for _ in range(warmup_iters):
            fn(logits_fp32, top_k=top_k, bias_fp32=bias_fp32)
        if is_cuda:
            torch.cuda.synchronize(logits_fp32.device)
        start = time.perf_counter()
        for _ in range(measure_iters):
            fn(logits_fp32, top_k=top_k, bias_fp32=bias_fp32)
        if is_cuda:
            torch.cuda.synchronize(logits_fp32.device)
    elapsed_s = time.perf_counter() - start
    return (elapsed_s * 1000.0) / measure_iters


def _run_case(
    device: torch.device,
    *,
    case_name: str,
    batch_size: int,
    seq_len: int,
    num_fine_experts: int,
    top_k: int,
) -> None:
    seed = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    logits_fp32 = torch.randn(
        batch_size,
        seq_len,
        num_fine_experts,
        generator=generator,
        dtype=torch.float32,
    ).to(device)

    eager_scores = _router_scores_training(logits_fp32, top_k=top_k)
    export_scores = _router_scores_export_safe(logits_fp32, top_k=top_k)
    max_abs_diff = (eager_scores - export_scores).abs().max().item()

    eager_ms = _benchmark(
        _router_scores_training,
        logits_fp32,
        top_k=top_k,
    )
    export_ms = _benchmark(
        _router_scores_export_safe,
        logits_fp32,
        top_k=top_k,
    )
    delta_pct = ((export_ms - eager_ms) / eager_ms) * 100.0

    print(
        f"{device.type}:{case_name}: "
        f"shape=({batch_size}, {seq_len}, {num_fine_experts}), "
        f"top_k={top_k}, "
        f"training={eager_ms:.6f} ms, "
        f"export_safe={export_ms:.6f} ms, "
        f"delta={delta_pct:.2f}%, "
        f"max_abs_diff={max_abs_diff:.3e}"
    )


def main() -> None:
    shape_cfg = _load_b0310_shape_config()
    num_fine_experts = shape_cfg["num_fine_experts"]
    top_k = shape_cfg["top_k"]
    max_ctx_len = shape_cfg["max_ctx_len"]

    cases = [
        ("single_step_export", 1, 1),
        ("training_like_sequence", 16, max_ctx_len),
    ]
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    print(
        "Benchmarking MoE router formulas with "
        f"num_fine_experts={num_fine_experts}, top_k={top_k}, "
        f"max_ctx_len={max_ctx_len}"
    )
    for device in devices:
        for case_name, batch_size, seq_len in cases:
            _run_case(
                device,
                case_name=case_name,
                batch_size=batch_size,
                seq_len=seq_len,
                num_fine_experts=num_fine_experts,
                top_k=top_k,
            )


if __name__ == "__main__":
    main()
