import sys
import tempfile
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


class _ExportAttentionModule(nn.Module):
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


class _ExportCausalAttentionModule(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        return export_safe_scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )


def _export_model(
    export_path: Path,
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
) -> None:
    torch.onnx.export(
        module.eval(),
        inputs,
        str(export_path),
        opset_version=17,
        input_names=input_names,
        output_names=["out"],
        dynamo=False,
        verbose=False,
    )


def _export_op_types(
    module: nn.Module,
    *inputs: torch.Tensor,
    input_names: list[str],
) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        export_path = Path(tmp_dir) / "attention.onnx"
        _export_model(export_path, module, inputs, input_names)
        model = onnx.load(str(export_path))
    return [node.op_type for node in model.graph.node]


def _run_onnx(
    module: nn.Module,
    *inputs: torch.Tensor,
    input_names: list[str],
) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp_dir:
        export_path = Path(tmp_dir) / "attention.onnx"
        _export_model(export_path, module, inputs, input_names)
        session = onnxruntime.InferenceSession(
            str(export_path),
            providers=["CPUExecutionProvider"],
        )
        feed = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(input_names, inputs, strict=True)
        }
        outputs = session.run(["out"], feed)
    return outputs[0]


def test_export_safe_attention_uses_native_bool_mask_outside_export(
    monkeypatch,
):
    captured = {}
    original_sdpa = F.scaled_dot_product_attention

    def _spy_sdpa(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        captured["mask_dtype"] = None if attn_mask is None else attn_mask.dtype
        return original_sdpa(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )

    monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: False)
    monkeypatch.setattr(F, "scaled_dot_product_attention", _spy_sdpa)

    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 5, 4)
    v = torch.randn(1, 2, 5, 4)
    mask = torch.ones(1, 1, 3, 5, dtype=torch.bool)

    export_safe_scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    )

    assert captured["mask_dtype"] == torch.bool


def test_export_safe_attention_matches_sdpa_for_valid_masks():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 3, 8)
    k = torch.randn(2, 4, 5, 8)
    v = torch.randn(2, 4, 5, 8)
    mask = torch.tensor(
        [
            [[[True, True, False, False, False]]],
            [[[True, False, False, False, False]]],
        ],
        dtype=torch.bool,
    ).expand(2, 1, 3, 5)

    expected = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    )
    actual = export_safe_scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    )

    torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=1.0e-5)


def test_legacy_attention_export_avoids_isnan():
    torch.manual_seed(2)
    q = torch.randn(1, 4, 1, 8)
    k = torch.randn(1, 4, 16, 8)
    v = torch.randn(1, 4, 16, 8)
    mask = torch.ones(1, 1, 1, 16, dtype=torch.bool)

    op_types = _export_op_types(
        _ExportAttentionModule(),
        q,
        k,
        v,
        mask,
        input_names=["q", "k", "v", "mask"],
    )

    assert "IsNaN" not in op_types


def test_legacy_attention_export_ort_matches_pytorch_for_future_mask():
    torch.manual_seed(3)
    q = torch.randn(2, 4, 3, 8)
    k = torch.randn(2, 4, 5, 8)
    v = torch.randn(2, 4, 5, 8)
    mask = torch.tensor(
        [
            [[[True, True, True, False, False]]],
            [[[True, False, False, False, False]]],
        ],
        dtype=torch.bool,
    ).expand(2, 1, 3, 5)

    expected = export_safe_scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    )
    actual = _run_onnx(
        _ExportAttentionModule(),
        q,
        k,
        v,
        mask,
        input_names=["q", "k", "v", "mask"],
    )

    np.testing.assert_allclose(
        actual, expected.detach().cpu().numpy(), atol=1.0e-6, rtol=1.0e-5
    )


def test_legacy_attention_export_ort_matches_pytorch_for_causal_path():
    torch.manual_seed(4)
    q = torch.randn(2, 4, 6, 8)
    k = torch.randn(2, 4, 6, 8)
    v = torch.randn(2, 4, 6, 8)

    expected = export_safe_scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
    )
    actual = _run_onnx(
        _ExportCausalAttentionModule(),
        q,
        k,
        v,
        input_names=["q", "k", "v"],
    )

    np.testing.assert_allclose(
        actual, expected.detach().cpu().numpy(), atol=1.0e-6, rtol=1.0e-5
    )


def test_legacy_attention_export_ort_matches_pytorch_for_kv_mask():
    torch.manual_seed(5)
    q = torch.randn(2, 4, 1, 8)
    k = torch.randn(2, 4, 16, 8)
    v = torch.randn(2, 4, 16, 8)
    valid_lengths = torch.tensor([16, 5], dtype=torch.int64)
    mask = (
        torch.arange(16, dtype=torch.int64)[None, :] < valid_lengths[:, None]
    )
    mask = mask[:, None, None, :]

    expected = export_safe_scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    )
    actual = _run_onnx(
        _ExportAttentionModule(),
        q,
        k,
        v,
        mask,
        input_names=["q", "k", "v", "mask"],
    )

    np.testing.assert_allclose(
        actual, expected.detach().cpu().numpy(), atol=1.0e-6, rtol=1.0e-5
    )
