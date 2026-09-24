"""Registry and independent CPU checks. Standard library only."""

import math

from learning.checks import close, equal, unchanged

WORKSHOPS = {
    "strided_softmax": (
        "intermediate",
        "GPU",
        "Load row*stride(0)+col*stride(1); store row*N+col.",
    ),
    "softmax_backward": ("intermediate", "GPU", "dx=y*(g-sum(y*g)); accumulate the dot in FP32."),
    "fused_gemm": ("intermediate", "GPU", "Mask M/N/K tails; add bias and ReLU before casting."),
    "online_normalizer": ("advanced", "CPU", "Merge at max(m1,m2); rescale BOTH denominators."),
    "split_softmax": (
        "advanced",
        "GPU",
        "Write partial (m,l), merge with exp(m_part-m), reread x.",
    ),
    "streaming_attention": ("advanced", "GPU", "When the running max changes, rescale l AND acc."),
}


def cpu_cases(name):
    if name != "online_normalizer":
        raise ValueError(f"No CPU checks registered for {name}")
    empty = (-math.inf, 0.0)

    def partitions(module):
        values = [-1000, -999, 0, 1000, 999, 1000, -700]
        oracle = [math.exp(x - 1000) for x in values]
        oracle = [v / math.fsum(oracle) for v in oracle]
        for size in (1, 2, 3, 32):
            close(module.streaming_softmax(values, size), oracle)

    def associativity(module):
        a, b, c = (0.0, 2.0), (1000.0, 1.0), (999.0, 3.0)
        expected = (1000, 1 + 3 / math.e)
        close(module.merge(module.merge(a, b), c), expected)
        close(module.merge(a, module.merge(b, c)), expected)
        close(module.merge(c, module.merge(b, a)), expected)

    def invalid_chunk(module):
        for size in (0, -1, 1.5):
            try:
                module.streaming_softmax([1], size)
            except ValueError:
                continue
            raise AssertionError(f"Expected ValueError for chunk_size={size}")

    return [
        ("empty summary identity", lambda m: equal(m.summarize([]), empty)),
        ("worked summary", lambda m: close(m.summarize([2, 2]), (2, 2))),
        ("empty/empty avoids NaN", lambda m: equal(m.merge(empty, empty), empty)),
        ("left identity", lambda m: equal(m.merge(empty, (3, 2)), (3, 2))),
        ("right identity", lambda m: equal(m.merge((3, 2), empty), (3, 2))),
        ("rescale old sum", lambda m: close(m.merge((2, 2), (4, 1)), (4, 1 + 2 / math.e**2))),
        ("partition invariance and overflow resistance", partitions),
        ("merge tree and order invariance", associativity),
        ("empty probabilities", lambda m: equal(m.streaming_softmax([], 3), [])),
        ("input unchanged", lambda m: unchanged(m.streaming_softmax, [1, 2, 3], 2)),
        ("invalid chunk size rejected", invalid_chunk),
    ]
