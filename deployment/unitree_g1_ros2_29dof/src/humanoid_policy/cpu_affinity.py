"""CPU affinity helpers used by the 29DOF deployment runtime."""

from __future__ import annotations


def parse_cpu_affinity(value) -> list[int]:
    """Parse '0,1' or '2' into CPU IDs. Empty or invalid entries are ignored."""
    text = str(value).strip()
    if not text:
        return []
    cpu_ids = []
    for item in text.split(","):
        item = item.strip()
        if item.isdigit():
            cpu_ids.append(int(item))
    return cpu_ids


def set_thread_cpu_affinity(cpu_ids: list[int]) -> bool:
    """Pin the current Linux thread to the given CPU IDs."""
    if not cpu_ids:
        return False
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        cpu_set_size = 1024
        n_cpu_bits = 8 * ctypes.sizeof(ctypes.c_ulong)
        n_longs = (cpu_set_size + n_cpu_bits - 1) // n_cpu_bits

        class CpuSetT(ctypes.Structure):
            _fields_ = [("__bits", ctypes.c_ulong * n_longs)]

        libc.pthread_self.restype = ctypes.c_ulong
        libc.pthread_setaffinity_np.argtypes = [
            ctypes.c_ulong,
            ctypes.c_size_t,
            ctypes.POINTER(CpuSetT),
        ]
        cpu_set = CpuSetT()
        for index in range(n_longs):
            cpu_set.__bits[index] = 0
        for cpu_id in cpu_ids:
            if 0 <= cpu_id < cpu_set_size:
                index = cpu_id // n_cpu_bits
                bit = cpu_id % n_cpu_bits
                cpu_set.__bits[index] |= 1 << bit
        thread_id = libc.pthread_self()
        size = ctypes.sizeof(CpuSetT)
        return libc.pthread_setaffinity_np(thread_id, size, ctypes.byref(cpu_set)) == 0
    except Exception:
        return False
