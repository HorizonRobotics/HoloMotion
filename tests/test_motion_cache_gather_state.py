import sys
import unittest
from unittest import mock
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holomotion.src.training.h5_dataloader import (
    ClipBatch,
    Hdf5RootDofDataset,
    MotionClipBatchCache,
    MotionWindow,
    _CpuFKTransform,
    _WorldFrameNormalizeTransform,
    build_motion_datasets_from_cfg,
)


def _expected_field(
    tensor: torch.Tensor,
    clip_indices: torch.Tensor,
    frame_indices: torch.Tensor,
    n_future_frames: int,
    lengths: torch.Tensor,
) -> torch.Tensor:
    temporal_span = 1 + int(n_future_frames)
    time_offsets = torch.arange(temporal_span, dtype=torch.long)
    gather_timesteps = frame_indices[:, None] + time_offsets[None, :]
    max_valid = torch.clamp(lengths.index_select(0, clip_indices) - 1, min=0)
    gather_timesteps = torch.minimum(gather_timesteps, max_valid[:, None])
    return tensor[clip_indices[:, None], gather_timesteps]


class MotionCacheGatherStateTests(unittest.TestCase):
    def test_build_motion_datasets_uses_only_raw_reference_prefix(self):
        with (
            mock.patch(
                "holomotion.src.training.h5_dataloader.preview_sampling_from_cfg"
            ),
            mock.patch(
                "holomotion.src.training.h5_dataloader.Hdf5RootDofDataset"
            ) as dataset_cls,
        ):
            build_motion_datasets_from_cfg(
                {
                    "backend": "hdf5_v2",
                    "hdf5_root": "/tmp/train",
                    "fk_robot_file_path": "robot.xml",
                },
                max_frame_length=16,
                min_window_length=4,
            )

        self.assertEqual(dataset_cls.call_count, 1)
        self.assertEqual(dataset_cls.call_args.kwargs["allowed_prefixes"], ["ref_"])

    def test_gather_tensor_returns_expected_values(self):
        cache = MotionClipBatchCache.__new__(MotionClipBatchCache)

        ref_dof_pos = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(
            2, 6, 3
        )
        ref_rg_pos = torch.arange(2 * 6 * 2 * 3, dtype=torch.float32).reshape(
            2, 6, 2, 3
        )
        lengths = torch.tensor([6, 4], dtype=torch.long)
        window_indices = torch.tensor([10, 11], dtype=torch.long)

        cache._current_batch = ClipBatch(
            tensors={
                "ref_dof_pos": ref_dof_pos,
                "ref_rg_pos": ref_rg_pos,
            },
            lengths=lengths,
            motion_keys=["clip-a", "clip-b"],
            raw_motion_keys=["clip-a", "clip-b"],
            window_indices=window_indices,
            max_frame_length=6,
        )

        clip_indices = torch.tensor([1, 0, 1, 1], dtype=torch.long)
        frame_indices = torch.tensor([0, 2, 3, 1], dtype=torch.long)

        gathered_dof_pos = cache.gather_tensor(
            "ref_dof_pos",
            clip_indices=clip_indices,
            frame_indices=frame_indices,
            n_future_frames=2,
        )
        gathered_rg_pos = cache.gather_tensor(
            "ref_rg_pos",
            clip_indices=clip_indices,
            frame_indices=frame_indices,
            n_future_frames=2,
        )

        expected_dof_pos = _expected_field(
            ref_dof_pos,
            clip_indices,
            frame_indices,
            n_future_frames=2,
            lengths=lengths,
        )
        expected_rg_pos = _expected_field(
            ref_rg_pos,
            clip_indices,
            frame_indices,
            n_future_frames=2,
            lengths=lengths,
        )

        torch.testing.assert_close(gathered_dof_pos, expected_dof_pos)
        torch.testing.assert_close(gathered_rg_pos, expected_rg_pos)
        self.assertEqual(tuple(gathered_dof_pos.shape), (4, 3, 3))
        self.assertEqual(tuple(gathered_rg_pos.shape), (4, 3, 2, 3))

    def test_gather_tensor_reflects_updated_indices_without_cached_state(self):
        cache = MotionClipBatchCache.__new__(MotionClipBatchCache)

        ref_dof_pos = torch.arange(3 * 6 * 3, dtype=torch.float32).reshape(
            3, 6, 3
        )
        lengths = torch.tensor([6, 5, 4], dtype=torch.long)
        window_indices = torch.tensor([10, 11, 12], dtype=torch.long)

        cache._current_batch = ClipBatch(
            tensors={"ref_dof_pos": ref_dof_pos},
            lengths=lengths,
            motion_keys=["clip-a", "clip-b", "clip-c"],
            raw_motion_keys=["clip-a", "clip-b", "clip-c"],
            window_indices=window_indices,
            max_frame_length=6,
        )

        initial_clip_indices = torch.tensor([0, 1, 2, 1], dtype=torch.long)
        initial_frame_indices = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        updated_clip_indices = torch.tensor([0, 2, 1, 0], dtype=torch.long)
        updated_frame_indices = torch.tensor([1, 0, 3, 2], dtype=torch.long)

        initial_gathered = cache.gather_tensor(
            "ref_dof_pos",
            clip_indices=initial_clip_indices,
            frame_indices=initial_frame_indices,
            n_future_frames=2,
        )
        updated_gathered = cache.gather_tensor(
            "ref_dof_pos",
            clip_indices=updated_clip_indices,
            frame_indices=updated_frame_indices,
            n_future_frames=2,
        )

        expected_initial = _expected_field(
            ref_dof_pos,
            initial_clip_indices,
            initial_frame_indices,
            n_future_frames=2,
            lengths=lengths,
        )
        expected_updated = _expected_field(
            ref_dof_pos,
            updated_clip_indices,
            updated_frame_indices,
            n_future_frames=2,
            lengths=lengths,
        )

        torch.testing.assert_close(initial_gathered, expected_initial)
        torch.testing.assert_close(updated_gathered, expected_updated)

    def test_cpu_fk_transform_does_not_pass_smoothing_configuration(self):
        transform = _CpuFKTransform.__new__(_CpuFKTransform)
        transform._fk = mock.Mock(
            return_value={
                "global_translation": torch.zeros(1, 4, 2, 3),
                "global_rotation_quat": torch.zeros(1, 4, 2, 4),
                "global_velocity": torch.zeros(1, 4, 2, 3),
                "global_angular_velocity": torch.zeros(1, 4, 2, 3),
                "dof_vel": torch.zeros(1, 4, 2),
            }
        )
        arrays = {
            "ref_root_pos": torch.zeros(4, 3),
            "ref_root_rot": torch.zeros(4, 4),
            "ref_dof_pos": torch.zeros(4, 2),
        }

        transform(arrays, fps=60.0, prefix="ref_")

        self.assertNotIn("vel_smoothing_sigma", transform._fk.call_args.kwargs)

    def test_hdf5_v2_sample_contains_no_filtered_reference(self):
        sample = self._make_stub_root_dof_dataset()[0]
        self.assertFalse(any(name.startswith("ft_ref_") for name in sample.tensors))
        self.assertNotIn("filter_cutoff_hz", sample.tensors)

    @staticmethod
    def _make_stub_root_dof_dataset(
        *,
        allowed_prefixes=("ref_",),
    ):
        dataset = Hdf5RootDofDataset.__new__(Hdf5RootDofDataset)
        dataset.windows = [
            MotionWindow(
                motion_key="clip-a__start_0_len_4",
                shard_index=0,
                start=0,
                length=4,
                raw_motion_key="clip-a",
                window_index=0,
            )
        ]
        dataset.clips = {
            "clip-a": {
                "metadata": {
                    "motion_fps": 60.0,
                }
            }
        }
        dataset._progress_counter = None
        dataset._world_frame_transform = _WorldFrameNormalizeTransform()
        dataset._file_handles = {}
        dataset._h5_access_counter = 0
        dataset._h5_cleanup_interval = int(1e6)
        dataset._allowed_prefixes = tuple(allowed_prefixes)
        dataset._fk_calls = []

        shard_handle = {
            "ref_root_pos": torch.arange(12, dtype=torch.float32)
            .reshape(4, 3)
            .numpy(),
            "ref_root_rot": torch.tensor(
                [[0.0, 0.0, 0.0, 1.0]] * 4, dtype=torch.float32
            ).numpy(),
            "ref_dof_pos": torch.arange(8, dtype=torch.float32)
            .reshape(4, 2)
            .numpy(),
        }

        def fake_fk_transform(
            arrays,
            fps,
            prefix="ref_",
        ):
            del fps
            dataset._fk_calls.append(prefix)
            root_pos = arrays[f"{prefix}root_pos"]
            root_rot = arrays[f"{prefix}root_rot"]
            arrays[f"{prefix}rg_pos"] = torch.stack(
                [root_pos, root_pos], dim=1
            )
            arrays[f"{prefix}rb_rot"] = torch.stack(
                [root_rot, root_rot], dim=1
            )
            arrays[f"{prefix}body_vel"] = torch.zeros(
                4, 2, 3, dtype=torch.float32
            )
            arrays[f"{prefix}body_ang_vel"] = torch.zeros(
                4, 2, 3, dtype=torch.float32
            )
            arrays[f"{prefix}dof_vel"] = torch.zeros(4, 2, dtype=torch.float32)

        dataset._fk_transform = fake_fk_transform
        dataset._get_shard_handle = lambda shard_index: shard_handle
        return dataset
