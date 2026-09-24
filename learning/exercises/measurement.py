"""Lesson 7. Units and baselines are part of correctness."""


def speedup(baseline_ms, candidate_ms):
    """Return baseline / candidate. Example: 10 ms -> 5 ms means 2x."""
    raise NotImplementedError("Use matching units and put the baseline in the numerator")


def effective_gbps(logical_bytes, elapsed_ms):
    """Return decimal GB/s (1 GB = 1e9 bytes), not GiB/s.

    This is useful bytes / time, not a measured DRAM counter.
    Example: 1_000_000 bytes in 1 ms equals 1 GB/s.
    """
    raise NotImplementedError("Convert milliseconds to seconds, then bytes to gigabytes")


def overall_speedup(fraction, kernel_speedup):
    """Amdahl's law for fraction in [0,1] and positive kernel_speedup.

    Normalize old program time to 1. Unchanged time is 1-fraction.
    Only the optimized fraction gets divided by kernel_speedup.
    """
    raise NotImplementedError("Compute the new total time, then divide old time by it")
