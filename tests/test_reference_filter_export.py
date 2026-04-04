import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holomotion.src.training.h5_dataloader import MotionClipSample
from holomotion.src.training.reference_filter_export import (
    export_reference_filter_artifacts_from_config,
    export_reference_filter_debug_artifacts,
)


def _quat_xyzw_from_rpy(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return torch.tensor(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=torch.float32,
    )


def _make_sample(*, include_filtered: bool = True) -> MotionClipSample:
    timesteps = 4
    num_bodies = 5
    num_dofs = 3

    ref_rg_pos = torch.arange(
        timesteps * num_bodies * 3, dtype=torch.float32
    ).reshape(timesteps, num_bodies, 3)
    ref_body_vel = ref_rg_pos + 100.0
    ref_body_ang_vel = ref_rg_pos + 200.0
    ref_dof_pos = torch.arange(
        timesteps * num_dofs, dtype=torch.float32
    ).reshape(timesteps, num_dofs)
    ref_dof_vel = ref_dof_pos + 50.0

    ref_rb_rot = torch.stack(
        [
            _quat_xyzw_from_rpy(0.0, 0.0, 0.0),
            _quat_xyzw_from_rpy(0.1, -0.2, 0.3),
            _quat_xyzw_from_rpy(0.2, -0.1, 0.4),
            _quat_xyzw_from_rpy(0.3, 0.0, 0.5),
        ],
        dim=0,
    )[:, None, :].repeat(1, num_bodies, 1)

    tensors = {
        "ref_rg_pos": ref_rg_pos,
        "ref_rb_rot": ref_rb_rot,
        "ref_body_vel": ref_body_vel,
        "ref_body_ang_vel": ref_body_ang_vel,
        "ref_root_pos": ref_rg_pos[:, 0, :],
        "ref_root_rot": ref_rb_rot[:, 0, :],
        "ref_root_vel": ref_body_vel[:, 0, :],
        "ref_root_ang_vel": ref_body_ang_vel[:, 0, :],
        "ref_dof_pos": ref_dof_pos,
        "ref_dof_vel": ref_dof_vel,
        "filter_cutoff_hz": torch.full(
            (timesteps, 1), 2.0, dtype=torch.float32
        ),
    }
    if include_filtered:
        tensors.update(
            {
                "ft_ref_rg_pos": ref_rg_pos + 0.5,
                "ft_ref_rb_rot": ref_rb_rot.clone(),
                "ft_ref_body_vel": ref_body_vel + 0.25,
                "ft_ref_body_ang_vel": ref_body_ang_vel + 0.25,
                "ft_ref_root_pos": ref_rg_pos[:, 0, :] + 0.5,
                "ft_ref_root_rot": ref_rb_rot[:, 0, :].clone(),
                "ft_ref_root_vel": ref_body_vel[:, 0, :] + 0.25,
                "ft_ref_root_ang_vel": ref_body_ang_vel[:, 0, :] + 0.25,
                "ft_ref_dof_pos": ref_dof_pos + 0.75,
                "ft_ref_dof_vel": ref_dof_vel + 0.75,
            }
        )

    return MotionClipSample(
        motion_key="clip-a__start_0_len_4",
        raw_motion_key="clip-a",
        window_index=0,
        tensors=tensors,
        length=timesteps,
    )


class ReferenceFilterExportTests(unittest.TestCase):
    def test_export_reference_filter_artifacts_from_config_builds_dataset(
        self,
    ):
        sample = _make_sample()
        body_names = [
            "root_link",
            "torso_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
            "left_ankle_roll_link",
        ]
        dof_names = [
            "waist_yaw_joint",
            "left_wrist_yaw_joint",
            "left_ankle_roll_joint",
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = OmegaConf.create(
                {
                    "robot": {
                        "body_names": body_names,
                        "dof_names": dof_names,
                        "motion": {
                            "online_filter": {"enabled": True},
                            "max_frame_length": 4,
                            "min_frame_length": 1,
                            "world_frame_normalization": True,
                        },
                    },
                    "debug_reference_filter_export": {
                        "enabled": True,
                        "output_dir": tmp_dir,
                        "selected_body_links": [
                            "left_wrist_yaw_link",
                            "left_ankle_roll_link",
                        ],
                    },
                }
            )

            with mock.patch(
                "holomotion.src.training.reference_filter_export."
                "build_motion_datasets_from_cfg",
                return_value=([sample], None, {}),
            ) as build_mock:
                output_dir = export_reference_filter_artifacts_from_config(
                    config
                )

            self.assertEqual(output_dir, Path(tmp_dir))
            build_mock.assert_called_once()
            self.assertTrue((Path(tmp_dir) / "metadata.json").is_file())

    def test_export_reference_filter_debug_artifacts_writes_outputs(self):
        sample = _make_sample()
        body_names = [
            "root_link",
            "torso_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
            "left_ankle_roll_link",
        ]
        dof_names = [
            "waist_yaw_joint",
            "left_wrist_yaw_joint",
            "left_ankle_roll_joint",
        ]
        selected_body_links = [
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
            "left_ankle_roll_link",
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            export_reference_filter_debug_artifacts(
                sample=sample,
                output_dir=output_dir,
                body_names=body_names,
                dof_names=dof_names,
                selected_body_links=selected_body_links,
            )

            self.assertTrue((output_dir / "metadata.json").is_file())
            self.assertTrue((output_dir / "root_signals.npz").is_file())
            self.assertTrue((output_dir / "bodylink_signals.npz").is_file())
            self.assertTrue((output_dir / "dof_signals.npz").is_file())
            self.assertTrue((output_dir / "root_comparison.png").is_file())
            self.assertTrue(
                (output_dir / "left_wrist_yaw_link_comparison.png").is_file()
            )
            self.assertTrue((output_dir / "dof_pos_comparison.png").is_file())
            self.assertTrue((output_dir / "dof_vel_comparison.png").is_file())

            metadata = json.loads(
                (output_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["filter_cutoff_hz"], 2.0)
            self.assertEqual(
                metadata["selected_body_links"], selected_body_links
            )
            self.assertEqual(metadata["dof_names"], dof_names)

            root_payload = np.load(output_dir / "root_signals.npz")
            self.assertIn("ref_global_pos", root_payload.files)
            self.assertIn("ft_ref_rpy", root_payload.files)
            self.assertEqual(root_payload["ref_global_pos"].shape, (4, 3))
            self.assertEqual(root_payload["ft_ref_rpy"].shape, (4, 3))

            dof_payload = np.load(output_dir / "dof_signals.npz")
            self.assertEqual(dof_payload["ref_dof_pos"].shape, (4, 3))
            self.assertEqual(dof_payload["ft_ref_dof_vel"].shape, (4, 3))

    def test_export_reference_filter_debug_artifacts_requires_filtered_tensors(
        self,
    ):
        sample = _make_sample(include_filtered=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(
                ValueError, "Filtered reference tensors are unavailable"
            ):
                export_reference_filter_debug_artifacts(
                    sample=sample,
                    output_dir=Path(tmp_dir),
                    body_names=["root_link", "left_wrist_yaw_link"],
                    dof_names=["waist_yaw_joint"],
                    selected_body_links=["left_wrist_yaw_link"],
                )
