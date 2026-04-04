from pathlib import Path
import sys
from types import SimpleNamespace

import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holomotion.src.algo.algo_base import BaseOnpolicyRL
import holomotion.src.training.h5_dataloader as h5_dataloader_module
from holomotion.src.training.h5_dataloader import MotionClipBatchCache


class _FakeDataset:
    def __init__(self, length: int = 8) -> None:
        self._length = int(length)
        self.max_frame_length = 16
        self.progress_counter = None

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        raise AssertionError("__getitem__ should not be called in these tests")

    def set_progress_counter(self, counter) -> None:
        self.progress_counter = counter

    def close(self) -> None:
        return


class MotionCacheStartupTests(unittest.TestCase):
    def test_motion_cache_uses_explicit_constructor_seed(self):
        with (
            mock.patch.object(
                MotionClipBatchCache, "_build_dataloader", lambda self: None
            ),
            mock.patch.object(
                MotionClipBatchCache, "_prime_buffers", lambda self: None
            ),
        ):
            cache = MotionClipBatchCache(
                train_dataset=_FakeDataset(),
                batch_size=2,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                seed=1234,
            )

        self.assertEqual(cache._seed, 1234)

    def test_setup_seeding_does_not_reinitialize_motion_cache(self):
        algo = BaseOnpolicyRL.__new__(BaseOnpolicyRL)
        algo.config = {"seed": 100}
        algo.process_rank = 2
        algo.command_name = "ref_motion"

        env_seed_calls = []
        motion_cache_seed_calls = []
        algo.env = SimpleNamespace(
            seed=lambda seed: env_seed_calls.append(seed)
        )
        algo.command_term = SimpleNamespace(
            cfg=SimpleNamespace(seed=102),
            set_motion_cache_seed=lambda seed,
            reinitialize: motion_cache_seed_calls.append((seed, reinitialize)),
        )

        BaseOnpolicyRL._setup_seeding(algo)

        self.assertEqual(algo.base_seed, 100)
        self.assertEqual(algo.seed, 102)
        self.assertEqual(env_seed_calls, [102])
        self.assertEqual(motion_cache_seed_calls, [(102, False)])

    def test_motion_cache_passes_loader_timeout_to_dataloader(self):
        captured_kwargs = {}

        class _FakeLoader:
            def __init__(self, *args, **kwargs) -> None:
                del args
                captured_kwargs.update(kwargs)

        with (
            mock.patch.object(h5_dataloader_module, "DataLoader", _FakeLoader),
            mock.patch.object(
                MotionClipBatchCache, "_prime_buffers", lambda self: None
            ),
        ):
            MotionClipBatchCache(
                train_dataset=_FakeDataset(),
                batch_size=2,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                loader_timeout=17,
            )

        self.assertEqual(captured_kwargs["timeout"], 17)

    def test_motion_cache_disables_progress_bar_in_distributed_runs(self):
        with (
            mock.patch.object(
                MotionClipBatchCache, "_build_dataloader", lambda self: None
            ),
            mock.patch.object(
                MotionClipBatchCache, "_prime_buffers", lambda self: None
            ),
        ):
            cache = MotionClipBatchCache(
                train_dataset=_FakeDataset(),
                batch_size=2,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                sampler_world_size=8,
                batch_progress_bar=True,
            )

        self.assertIs(cache._should_use_batch_progress(), False)
        self.assertIsNone(cache._batch_progress_counter)

    def test_motion_cache_keeps_progress_bar_for_local_runs(self):
        with (
            mock.patch.object(
                MotionClipBatchCache, "_build_dataloader", lambda self: None
            ),
            mock.patch.object(
                MotionClipBatchCache, "_prime_buffers", lambda self: None
            ),
        ):
            cache = MotionClipBatchCache(
                train_dataset=_FakeDataset(),
                batch_size=2,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                sampler_world_size=1,
                batch_progress_bar=True,
            )

        self.assertIs(cache._should_use_batch_progress(), True)
        self.assertIsNotNone(cache._batch_progress_counter)

    def test_motion_cache_requires_positive_loader_timeout(self):
        with (
            mock.patch.object(
                MotionClipBatchCache, "_build_dataloader", lambda self: None
            ),
            mock.patch.object(
                MotionClipBatchCache, "_prime_buffers", lambda self: None
            ),
        ):
            with self.assertRaisesRegex(
                ValueError, "loader_timeout must be >= 0"
            ):
                MotionClipBatchCache(
                    train_dataset=_FakeDataset(),
                    batch_size=2,
                    num_workers=0,
                    pin_memory=False,
                    persistent_workers=False,
                    loader_timeout=-1,
                )
