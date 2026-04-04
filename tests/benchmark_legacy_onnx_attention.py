import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holomotion.src.modules.network_modules import (
    export_safe_scaled_dot_product_attention,
)


class _RawAttentionModule(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )


class _SafeAttentionModule(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return export_safe_scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
        )


def _export_model(
    module: nn.Module,
    export_path: Path,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    torch.onnx.export(
        module.eval(),
        (q, k, v, mask),
        str(export_path),
        opset_version=17,
        input_names=["q", "k", "v", "mask"],
        output_names=["out"],
        dynamo=False,
        verbose=False,
    )


def _benchmark_session(
    model_path: Path,
    provider,
    feed: dict[str, np.ndarray],
    *,
    warmup_iters: int = 50,
    measure_iters: int = 300,
) -> float:
    providers = (
        ["CPUExecutionProvider"]
        if provider == "CPUExecutionProvider"
        else [provider, "CPUExecutionProvider"]
    )
    session = onnxruntime.InferenceSession(
        str(model_path),
        providers=providers,
    )
    for _ in range(warmup_iters):
        session.run(["out"], feed)
    start = time.perf_counter()
    for _ in range(measure_iters):
        session.run(["out"], feed)
    elapsed_s = time.perf_counter() - start
    return (elapsed_s * 1000.0) / measure_iters


def main() -> None:
    torch.manual_seed(0)
    q = torch.randn(4, 8, 1, 64)
    k = torch.randn(4, 8, 32, 64)
    v = torch.randn(4, 8, 32, 64)
    valid_lengths = torch.tensor([32, 24, 16, 8], dtype=torch.int64)
    mask = (
        torch.arange(32, dtype=torch.int64)[None, :] < valid_lengths[:, None]
    )
    mask = mask[:, None, None, :]
    feed = {
        "q": q.numpy(),
        "k": k.numpy(),
        "v": v.numpy(),
        "mask": mask.numpy(),
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_path = tmp_path / "raw_attention.onnx"
        safe_path = tmp_path / "safe_attention.onnx"
        _export_model(_RawAttentionModule(), raw_path, q, k, v, mask)
        _export_model(_SafeAttentionModule(), safe_path, q, k, v, mask)
        raw_model = onnx.load(str(raw_path))
        safe_model = onnx.load(str(safe_path))
        raw_ops = [node.op_type for node in raw_model.graph.node]
        safe_ops = [node.op_type for node in safe_model.graph.node]
        print(
            "Graph ops: "
            f"raw_has_isnan={'IsNaN' in raw_ops}, "
            f"safe_has_isnan={'IsNaN' in safe_ops}"
        )

        cpu_raw = _benchmark_session(raw_path, "CPUExecutionProvider", feed)
        cpu_safe = _benchmark_session(safe_path, "CPUExecutionProvider", feed)
        print(
            f"CPUExecutionProvider: raw={cpu_raw:.4f} ms, "
            f"safe={cpu_safe:.4f} ms, "
            f"delta={(cpu_safe - cpu_raw) / cpu_raw * 100.0:.2f}%"
        )

        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            cuda_raw = _benchmark_session(
                raw_path, "CUDAExecutionProvider", feed
            )
            cuda_safe = _benchmark_session(
                safe_path, "CUDAExecutionProvider", feed
            )
            print(
                f"CUDAExecutionProvider: raw={cuda_raw:.4f} ms, "
                f"safe={cuda_safe:.4f} ms, "
                f"delta={(cuda_safe - cuda_raw) / cuda_raw * 100.0:.2f}%"
            )


if __name__ == "__main__":
    main()
