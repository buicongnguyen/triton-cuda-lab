"""Reference answer: the denominator defines which performance claim is being made."""


def speedup(baseline_ms, candidate_ms):
    return baseline_ms / candidate_ms


def effective_gbps(logical_bytes, elapsed_ms):
    return logical_bytes / (elapsed_ms * 1e6)


def overall_speedup(fraction, kernel_speedup):
    return 1 / (1 - fraction + fraction / kernel_speedup)
