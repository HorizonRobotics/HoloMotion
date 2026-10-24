"""Return idle Torch allocations to independent physics allocators under pressure."""

import time

GIB = 1024**3


def cache_decision(free, total, reserved, allocated, external_growth=0):
    """Pure byte policy, including a conservative unobserved graph-peak floor.

    CUDA graph allocation peaks are invisible between launches. Until a backend
    supplies a measured bound, keep 25%–30% of device capacity available for them.
    Grow this floor up to 30% if the independent allocator's footprint grows.
    This is a capacity safety margin, not a claim to measure transient peaks.
    """
    reserve = max(2 * GIB, int(total * 0.25)) + min(
        external_growth // 8, int(total * 0.05)
    )
    reclaimable = max(0, reserved - allocated)
    return dict(
        release=free < reserve and reclaimable >= 256 * 1024**2,
        reserve_bytes=reserve,
        reclaimable_bytes=reclaimable,
        free_bytes=free,
    )


class PhysicsCachePressure:
    def __init__(self, torch, device, enabled=True, warmup_checks=64):
        self.torch, self.device, self.enabled = torch, device, enabled
        self.warmup_checks = warmup_checks
        self.external_floor = None
        self.external_high = 0
        self.checks = self.releases = 0
        self.check_seconds = self.release_seconds = 0.0
        self.minimum_free = None
        self.last = {}

    def before_launch(self):
        if not self.enabled:
            return
        started = time.monotonic()
        cuda = self.torch.cuda
        free, total = cuda.mem_get_info(self.device)
        reserved = cuda.memory_reserved(self.device)
        allocated = cuda.memory_allocated(self.device)
        external = max(0, total - free - reserved)
        if self.external_floor is None or self.checks < self.warmup_checks:
            # First physics graphs may grow their pools lazily. Establish the
            # independent-pool baseline only after these initial launches.
            self.external_floor = max(self.external_floor or 0, external)
        self.external_high = max(self.external_high, external)
        decision = cache_decision(
            free,
            total,
            reserved,
            allocated,
            self.external_high - self.external_floor,
        )
        self.checks += 1
        self.minimum_free = (
            free if self.minimum_free is None else min(self.minimum_free, free)
        )
        if decision["release"]:
            release_started = time.monotonic()
            # Scoped to the actual physics device; active tensors stay live.
            with cuda.device(self.device):
                cuda.empty_cache()
            self.release_seconds += time.monotonic() - release_started
            self.releases += 1
            decision["free_after_release_bytes"] = cuda.mem_get_info(
                self.device
            )[0]
        self.last = decision
        self.check_seconds += time.monotonic() - started

    def summary(self):
        return dict(
            policy="pre_physics_pressure_v1",
            enabled=self.enabled,
            checks=self.checks,
            releases=self.releases,
            check_seconds=self.check_seconds,
            release_seconds=self.release_seconds,
            minimum_precheck_free_bytes=self.minimum_free,
            external_persistent_high_bytes=self.external_high,
            external_baseline_bytes=self.external_floor,
            last=self.last,
        )
