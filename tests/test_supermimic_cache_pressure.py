from contextlib import contextmanager
from types import SimpleNamespace

from holomotion.src.algo.supermimic.cache_pressure import (
    GIB,
    PhysicsCachePressure,
    cache_decision,
)


def test_pressure_retains_cache_with_headroom():
    assert not cache_decision(12 * GIB, 24 * GIB, 6 * GIB, 4 * GIB)["release"]


def test_pressure_releases_idle_cache_before_graph_peak():
    d = cache_decision(5 * GIB, 24 * GIB, 15 * GIB, 5 * GIB)
    assert d["release"]
    assert d["reserve_bytes"] == int(24 * GIB * 0.25)
    assert d["reclaimable_bytes"] == 10 * GIB


def test_pressure_avoids_repeated_empty_without_idle_blocks():
    assert not cache_decision(GIB, 24 * GIB, 5 * GIB, 5 * GIB)["release"]
    assert not cache_decision(GIB, 24 * GIB, 5 * GIB + 1, 5 * GIB)["release"]


def test_pressure_adapts_to_independent_pool_growth():
    d = cache_decision(6 * GIB, 24 * GIB, 6 * GIB, 4 * GIB, 6 * GIB)
    assert d["release"]
    assert d["reserve_bytes"] == int(24 * GIB * 0.25) + 6 * GIB // 8


def test_actual_device_and_no_repeated_release_after_reclamation():
    class Cuda:
        free = 5 * GIB
        reserved = 15 * GIB
        calls = []

        def mem_get_info(self, device):
            self.calls.append(("query", device))
            return self.free, 24 * GIB

        def memory_reserved(self, device):
            assert device == "cuda:3"
            return self.reserved

        def memory_allocated(self, device):
            assert device == "cuda:3"
            return 5 * GIB

        @contextmanager
        def device(self, device):
            self.calls.append(("scope", device))
            yield

        def empty_cache(self):
            self.calls.append(("empty", "cuda:3"))
            self.free += 10 * GIB
            self.reserved -= 10 * GIB

    cuda = Cuda()
    p = PhysicsCachePressure(SimpleNamespace(cuda=cuda), "cuda:3")
    p.before_launch()
    p.before_launch()
    assert p.releases == 1 and p.checks == 2
    assert cuda.calls.count(("empty", "cuda:3")) == 1
    assert ("scope", "cuda:3") in cuda.calls
