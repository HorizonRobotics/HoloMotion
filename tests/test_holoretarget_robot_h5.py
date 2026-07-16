import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from holomotion.src.training.data_production.distributed_robot_h5 import (
    _read_optional_config,
    ClipTask,
    H5RootInfo,
    _is_holosmpl_h5_manifest,
    finalize_manifests,
)
from holomotion.src.motion_tracking.reference_observation import (
    derive_reference_kinematics_numpy,
)
from holomotion.src.training.data_production.robot_h5 import (
    ROBOT_H5_ARRAY_NAMES,
    RobotH5ShardWriter,
)
from holomotion.src.training.h5_dataloader import Hdf5RootDofDataset


class HoloRetargetRobotH5Tests(unittest.TestCase):
    def test_yaml_config_resolves_inherited_root_interpolation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "base.yaml").write_text(
                "holosmpl_run_root: /bucket/run\nsources: []\n",
                encoding="utf-8",
            )
            (root / "sources.yaml").write_text(
                "defaults:\n"
                "  - base\n"
                "  - _self_\n"
                "sources:\n"
                "  - holosmpl_h5_root: ${holosmpl_run_root}/train/final/formal_h5\n",
                encoding="utf-8",
            )

            config = _read_optional_config(root / "sources.yaml")

        self.assertEqual(
            config["sources"][0]["holosmpl_h5_root"],
            "/bucket/run/train/final/formal_h5",
        )

    def test_discovery_accepts_clip_level_shape_beta(self):
        manifest = {
            "shards": [],
            "available_arrays": ["human_pose_aa", "human_root_trans"],
            "clip_level_arrays": ["clips/human_shape_beta"],
        }
        self.assertTrue(_is_holosmpl_h5_manifest(manifest))

    def test_reference_kinematics_uses_sample_timestamps_and_boundaries(self):
        qpos = np.zeros((3, 36), dtype=np.float32)
        qpos[:, 3] = 1.0
        qpos[1, 7:] = 2.0
        qpos[2, 7:] = 8.0
        result = derive_reference_kinematics_numpy(
            qpos,
            sample_time=np.asarray([0.0, 2.0, 3.0], dtype=np.float32),
        )

        np.testing.assert_allclose(result.dof_vel[0], 1.0, atol=0.0)
        np.testing.assert_allclose(result.dof_vel[1], 8.0 / 3.0, atol=1.0e-6)
        np.testing.assert_allclose(result.dof_vel[2], 6.0, atol=0.0)

    def test_writer_persists_minimal_reference_schema(self):
        arrays = self._reference_arrays(frame_count=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shard.h5"
            writer = RobotH5ShardWriter(path, compression=None)
            writer.append_motion(
                motion_id=7,
                arrays=arrays,
                metadata_json='{"motion_fps": 50.0}',
            )
            summary = writer.finalize()

            self.assertEqual(summary["num_clips"], 1)
            self.assertEqual(summary["num_frames"], 4)
            with h5py.File(path, "r") as handle:
                self.assertEqual(
                    set(ROBOT_H5_ARRAY_NAMES), set(handle.keys()) - {"clips"}
                )
                for name in ROBOT_H5_ARRAY_NAMES:
                    np.testing.assert_array_equal(
                        handle[name][:], arrays[name]
                    )
                np.testing.assert_array_equal(handle["clips/start"][:], [0])
                np.testing.assert_array_equal(handle["clips/length"][:], [4])
                np.testing.assert_array_equal(
                    handle["clips/motion_key_id"][:], [7]
                )

    def test_hdf5_v2_loader_derives_state_from_minimal_schema(self):
        arrays = self._reference_arrays(frame_count=4)
        arrays["ref_root_pos"][:, 0] = np.arange(4, dtype=np.float32) * 0.02
        arrays["ref_dof_pos"][:] = (
            np.arange(4, dtype=np.float32)[:, None] * 0.02
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_path = root / "shards" / "shard.h5"
            writer = RobotH5ShardWriter(shard_path, compression=None)
            writer.append_motion(
                motion_id=0,
                arrays=arrays,
                metadata_json='{"motion_fps": 50.0}',
            )
            writer.finalize()
            manifest = {
                "hdf5_shards": [{"file": "shards/shard.h5"}],
                "clips": {
                    "clip": {
                        "shard": 0,
                        "start": 0,
                        "length": 4,
                        "metadata": {"motion_fps": 50.0},
                    }
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest))

            robot_file = (
                Path(__file__).resolve().parents[1]
                / "assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.urdf"
            )
            dataset = Hdf5RootDofDataset(
                manifest_path=str(root / "manifest.json"),
                max_frame_length=4,
                min_window_length=1,
                fk_robot_file_path=str(robot_file),
            )
            sample = dataset[0].tensors
            np.testing.assert_allclose(sample["ref_dof_vel"], 1.0, atol=1.0e-5)
            np.testing.assert_allclose(
                sample["ref_root_vel"][:, 0], 1.0, atol=1.0e-5
            )
            np.testing.assert_array_equal(
                sample["ref_root_pos"], arrays["ref_root_pos"]
            )
            dataset.close()

    def test_distributed_finalize_converts_shard_paths_to_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "output"
            dataset_root = output_root / "dataset" / "holoretarget_h5"
            coord_root = root / "coord"
            coord_root.mkdir()
            rank_result = {
                "roots": {
                    "dataset": {
                        "hdf5_shards": [
                            {"file": "shards/rank1.h5"},
                            {"file": "shards/rank0.h5"},
                        ],
                        "clips": {
                            "clip0": {"shard": "shards/rank0.h5"},
                            "clip1": {"shard": "shards/rank1.h5"},
                        },
                        "processed_clips": 2,
                        "processed_frames": 8,
                        "dof_names": ["joint"],
                    }
                }
            }
            (coord_root / "rank_00000.done.json").write_text(
                json.dumps(rank_result)
            )
            root_info = H5RootInfo(
                root_index=0,
                input_root=root / "input",
                output_root=dataset_root,
                relative_key="dataset",
                display_name="dataset",
                source_manifest={"schema_version": "holosmpl_v1"},
            )
            tasks = [
                ClipTask(
                    root_index=0,
                    global_index=index,
                    input_root=root / "input",
                    output_root=dataset_root,
                    shard_rel="source.h5",
                    clip_index=index,
                    start=index * 4,
                    length=4,
                    motion_key=f"clip{index}",
                    metadata={"motion_fps": 50.0},
                )
                for index in range(2)
            ]
            finalize_manifests(
                output_root=output_root,
                coord_root=coord_root,
                root_infos=[root_info],
                tasks=tasks,
                compression="lzf",
                chunks_t=1024,
            )
            manifest = json.loads((dataset_root / "manifest.json").read_text())
            self.assertEqual(
                [item["file"] for item in manifest["hdf5_shards"]],
                ["shards/rank0.h5", "shards/rank1.h5"],
            )
            self.assertEqual(manifest["clips"]["clip0"]["shard"], 0)
            self.assertEqual(manifest["clips"]["clip1"]["shard"], 1)

    @staticmethod
    def _reference_arrays(frame_count: int):
        root_rot = np.zeros((frame_count, 4), dtype=np.float32)
        root_rot[:, 3] = 1.0
        return {
            "ref_root_pos": np.zeros((frame_count, 3), dtype=np.float32),
            "ref_root_rot": root_rot,
            "ref_dof_pos": np.zeros((frame_count, 29), dtype=np.float32),
        }


if __name__ == "__main__":
    unittest.main()
