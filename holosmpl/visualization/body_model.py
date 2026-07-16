from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from holosmpl.visualization.canonical_loader import CanonicalClip


@dataclass(frozen=True)
class BodyModelOutput:
    vertices: np.ndarray
    joints: np.ndarray
    faces: np.ndarray
    model_type: str
    gender: str


class BodyModelCache:
    def __init__(self, models_root: str | Path, *, device: str = "cpu") -> None:
        self.models_root = Path(models_root)
        self.device = device
        self._models: dict[tuple[str, str, int], Any] = {}
        self._temporary_roots: list[tempfile.TemporaryDirectory[str]] = []

    def forward_clip_frames(
        self,
        clip: CanonicalClip,
        frame_indices: list[int],
        *,
        batch_size: int = 32,
    ) -> BodyModelOutput:
        import torch

        if not frame_indices:
            raise ValueError("frame_indices must not be empty")

        requested_model_type = model_type_from_layout(clip.pose_body_layout)
        gender = normalize_gender(clip.gender)
        model_type = requested_model_type
        pose_body_for_model = clip.pose_body
        output_model_type = model_type
        try:
            model = self._get_model(model_type, gender, int(clip.betas.shape[0]))
        except Exception:
            if requested_model_type != "smpl" or clip.pose_body.shape[1] != 69:
                raise
            model_type = "smplx"
            pose_body_for_model = clip.pose_body[:, :63]
            output_model_type = "smplx_fallback_for_smpl"
            model = self._get_model(model_type, gender, int(clip.betas.shape[0]))
        faces = np.asarray(model.faces, dtype=np.int32)

        vertices_chunks: list[np.ndarray] = []
        joints_chunks: list[np.ndarray] = []
        for start in range(0, len(frame_indices), batch_size):
            batch_indices = frame_indices[start : start + batch_size]
            global_orient = torch.tensor(
                clip.root_orient[batch_indices], dtype=torch.float32, device=self.device
            )
            body_pose = torch.tensor(
                pose_body_for_model[batch_indices], dtype=torch.float32, device=self.device
            )
            transl = torch.tensor(clip.trans[batch_indices], dtype=torch.float32, device=self.device)
            betas = torch.tensor(clip.betas[None, :], dtype=torch.float32, device=self.device)
            betas = betas.expand(len(batch_indices), -1).contiguous()

            kwargs: dict[str, Any] = {
                "global_orient": global_orient,
                "body_pose": body_pose,
                "betas": betas,
                "transl": transl,
                "return_verts": True,
            }
            if model_type == "smplx":
                batch = len(batch_indices)
                kwargs.update(
                    {
                        "jaw_pose": torch.zeros(batch, 3, dtype=torch.float32, device=self.device),
                        "leye_pose": torch.zeros(batch, 3, dtype=torch.float32, device=self.device),
                        "reye_pose": torch.zeros(batch, 3, dtype=torch.float32, device=self.device),
                        "left_hand_pose": torch.zeros(
                            batch, 45, dtype=torch.float32, device=self.device
                        ),
                        "right_hand_pose": torch.zeros(
                            batch, 45, dtype=torch.float32, device=self.device
                        ),
                        "expression": torch.zeros(
                            batch,
                            int(getattr(model, "num_expression_coeffs", 10)),
                            dtype=torch.float32,
                            device=self.device,
                        ),
                    }
                )
            with torch.no_grad():
                out = model(**kwargs)
            vertices_chunks.append(out.vertices.detach().cpu().numpy().astype(np.float32))
            joints_chunks.append(out.joints.detach().cpu().numpy().astype(np.float32))

        return BodyModelOutput(
            vertices=np.concatenate(vertices_chunks, axis=0),
            joints=np.concatenate(joints_chunks, axis=0),
            faces=faces,
            model_type=output_model_type,
            gender=gender,
        )

    def _get_model(self, model_type: str, gender: str, num_betas: int) -> Any:
        import inspect

        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
        _patch_numpy_legacy_aliases()
        import smplx

        key = (model_type, gender, num_betas)
        if key in self._models:
            return self._models[key]

        model_root = self._resolve_model_root(model_type)
        if not model_root.exists():
            raise FileNotFoundError(f"SMPL models root does not exist: {model_root}")

        errors: list[str] = []
        for candidate_gender in (gender, "neutral"):
            candidate_key = (model_type, candidate_gender, num_betas)
            if candidate_key in self._models:
                return self._models[candidate_key]
            try:
                model = smplx.create(
                    str(model_root),
                    model_type=model_type,
                    gender=candidate_gender,
                    num_betas=num_betas,
                    use_pca=False,
                    flat_hand_mean=True,
                    ext="pkl",
                ).to(self.device)
                model.eval()
            except Exception as exc:
                errors.append(f"{candidate_gender}: {type(exc).__name__}: {exc}")
                continue
            self._models[candidate_key] = model
            self._models[key] = model
            return model
        raise RuntimeError(
            f"failed to load {model_type} model under {model_root}: " + "; ".join(errors)
        )

    def _resolve_model_root(self, model_type: str) -> Path:
        if model_type != "smpl":
            return self.models_root
        if (self.models_root / "smpl").is_dir():
            return self.models_root

        legacy = (
            self.models_root.parent
            / "SMPL_python_v.1.1.0"
            / "smpl"
            / "models"
            / "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
        )
        if not legacy.is_file():
            return self.models_root

        temp_dir = tempfile.TemporaryDirectory(prefix="hsc_smpl_compat_")
        compat_root = Path(temp_dir.name)
        smpl_dir = compat_root / "smpl"
        smpl_dir.mkdir(parents=True, exist_ok=True)
        os.symlink(legacy, smpl_dir / "SMPL_NEUTRAL.pkl")
        self._temporary_roots.append(temp_dir)
        return compat_root


def normalize_gender(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"male", "female", "neutral"}:
        return text
    return "neutral"


def model_type_from_layout(pose_body_layout: str) -> str:
    if pose_body_layout == "smplx_21_body":
        return "smplx"
    if pose_body_layout == "smpl_23_body":
        return "smpl"
    raise ValueError(f"unsupported pose_body_layout for visualization: {pose_body_layout}")


def _patch_numpy_legacy_aliases() -> None:
    # Older SMPL/chumpy pickles still import removed NumPy scalar aliases.
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
