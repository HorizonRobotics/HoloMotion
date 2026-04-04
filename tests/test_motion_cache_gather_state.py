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
    _normalize_online_filter_cfg,
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
    def test_normalize_online_filter_cfg_includes_velocity_smoothing_sigmas(
        self,
    ):
        default_cfg = _normalize_online_filter_cfg({})

        self.assertEqual(default_cfg["ref_vel_smoothing_sigma"], 2.0)
        self.assertEqual(default_cfg["ft_ref_vel_smoothing_sigma"], 2.0)

        explicit_cfg = _normalize_online_filter_cfg(
            {
                "enabled": True,
                "butter_cutoff_hz_pool": [3.0],
                "ref_vel_smoothing_sigma": 0.0,
                "ft_ref_vel_smoothing_sigma": 2.0,
            },
            default_vel_smoothing_sigma=0.5,
        )

        self.assertEqual(explicit_cfg["ref_vel_smoothing_sigma"], 0.0)
        self.assertEqual(explicit_cfg["ft_ref_vel_smoothing_sigma"], 2.0)

    def test_normalize_online_filter_cfg_uses_fk_sigma_fallback_defaults(self):
        cfg = _normalize_online_filter_cfg(
            {},
            default_vel_smoothing_sigma=0.5,
        )

        self.assertEqual(cfg["ref_vel_smoothing_sigma"], 0.5)
        self.assertEqual(cfg["ft_ref_vel_smoothing_sigma"], 0.5)

    def test_build_motion_datasets_from_cfg_passes_fk_sigma_fallback(self):
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
                    "fk_vel_smoothing_sigma": 0.5,
                    "cache": {"allowed_prefixes": ["ref_", "ft_ref_"]},
                    "online_filter": {"enabled": False},
                },
                max_frame_length=16,
                min_window_length=4,
            )

        self.assertEqual(dataset_cls.call_count, 1)
        self.assertEqual(
            dataset_cls.call_args.kwargs["fk_vel_smoothing_sigma"],
            0.5,
        )

    def test_build_motion_datasets_from_cfg_defaults_fk_sigma_fallback(self):
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
                    "cache": {"allowed_prefixes": ["ref_", "ft_ref_"]},
                    "online_filter": {"enabled": False},
                },
                max_frame_length=16,
                min_window_length=4,
            )

        self.assertEqual(dataset_cls.call_count, 1)
        self.assertEqual(
            dataset_cls.call_args.kwargs["fk_vel_smoothing_sigma"],
            2.0,
        )

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

    def test_cpu_fk_transform_forwards_explicit_vel_smoothing_sigma(self):
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

        transform(
            arrays,
            fps=60.0,
            prefix="ref_",
            vel_smoothing_sigma=0.0,
        )

        self.assertEqual(
            transform._fk.call_args.kwargs["vel_smoothing_sigma"],
            0.0,
        )

    def test_cpu_fk_transform_defaults_vel_smoothing_sigma_to_two(self):
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

        transform(arrays, fps=60.0)

        self.assertEqual(
            transform._fk.call_args.kwargs["vel_smoothing_sigma"],
            2.0,
        )

    def test_hdf5_v2_sample_exposes_zero_cutoff_metadata_when_disabled(self):
        dataset = self._make_stub_root_dof_dataset()

        sample = dataset[0]

        self.assertIn("filter_cutoff_hz", sample.tensors)
        torch.testing.assert_close(
            sample.tensors["filter_cutoff_hz"],
            torch.zeros(4, 1, dtype=torch.float32),
        )

    def test_hdf5_v2_sample_exposes_sampled_cutoff_metadata(self):
        dataset = self._make_stub_root_dof_dataset(
            cutoff_pool=(3.0,),
            online_filter_enabled=True,
        )

        sample = dataset[0]

        self.assertIn("filter_cutoff_hz", sample.tensors)
        torch.testing.assert_close(
            sample.tensors["filter_cutoff_hz"],
            torch.full((4, 1), 3.0, dtype=torch.float32),
        )

    def test_hdf5_v2_sample_generates_filtered_reference_family(self):
        dataset = self._make_stub_root_dof_dataset(
            cutoff_pool=(3.0,),
            online_filter_enabled=True,
        )

        sample = dataset[0]

        for tensor_name in (
            "ft_ref_root_pos",
            "ft_ref_root_rot",
            "ft_ref_dof_pos",
            "ft_ref_rg_pos",
            "ft_ref_rb_rot",
            "ft_ref_body_vel",
            "ft_ref_body_ang_vel",
            "ft_ref_dof_vel",
            "ft_ref_root_vel",
            "ft_ref_root_ang_vel",
        ):
            self.assertIn(tensor_name, sample.tensors)

    def test_hdf5_v2_sample_uses_split_fk_smoothing_sigmas(self):
        dataset = self._make_stub_root_dof_dataset(
            cutoff_pool=(3.0,),
            online_filter_enabled=True,
            ref_vel_smoothing_sigma=0.0,
            ft_ref_vel_smoothing_sigma=2.0,
        )

        sample = dataset[0]

        self.assertIn("ref_root_vel", sample.tensors)
        self.assertIn("ft_ref_root_vel", sample.tensors)
        self.assertEqual(
            dataset._fk_calls,
            [("ref_", 0.0), ("ft_ref_", 2.0)],
        )

    def test_hdf5_v2_sample_skips_filtered_reference_family_when_disabled(
        self,
    ):
        dataset = self._make_stub_root_dof_dataset(
            cutoff_pool=(3.0,),
            online_filter_enabled=True,
            allowed_prefixes=("ref_",),
        )

        sample = dataset[0]

        self.assertNotIn("ft_ref_root_pos", sample.tensors)
        self.assertNotIn("ft_ref_rg_pos", sample.tensors)

    @staticmethod
    def _make_stub_root_dof_dataset(
        *,
        cutoff_pool=(0.0,),
        online_filter_enabled=False,
        allowed_prefixes=("ref_", "ft_ref_"),
        ref_vel_smoothing_sigma=2.0,
        ft_ref_vel_smoothing_sigma=2.0,
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
        dataset._world_frame_transform = None
        dataset._file_handles = {}
        dataset._h5_access_counter = 0
        dataset._h5_cleanup_interval = int(1e6)
        dataset._online_filter_enabled = bool(online_filter_enabled)
        dataset._online_filter_cutoff_hz_pool = tuple(cutoff_pool)
        dataset._allowed_prefixes = tuple(allowed_prefixes)
        dataset._ref_vel_smoothing_sigma = float(ref_vel_smoothing_sigma)
        dataset._ft_ref_vel_smoothing_sigma = float(ft_ref_vel_smoothing_sigma)
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

        dataset._online_filter_butter_order = 4

        def fake_fk_transform(
            arrays,
            fps,
            prefix="ref_",
            vel_smoothing_sigma=2.0,
        ):
            del fps
            dataset._fk_calls.append((prefix, float(vel_smoothing_sigma)))
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
