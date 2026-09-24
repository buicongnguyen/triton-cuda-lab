"""Concrete exercise checks. This module deliberately has no third-party imports."""

import copy
import math

CPU_EXERCISES = ("offsets", "strides", "reduction", "softmax", "rmsnorm", "matmul", "measurement")
GPU_EXERCISES = ("triton_add", "triton_sum", "triton_softmax")
HINTS = {
    "offsets": "ceil_div uses integer //; every program starts at program_id * block_size.",
    "strides": "storage_offset + row * row_stride + col * col_stride; strides count elements.",
    "reduction": "Zero leaves sums unchanged. Pad first; pair indices 0+1, 2+3, ... each round.",
    "softmax": "Subtract max(values) BEFORE math.exp. Divide by the SUM of exponentials.",
    "rmsnorm": "Square the residual sum, average over actual width, add eps, then take sqrt.",
    "matmul": "out[row][col] += a[row][inner] * b[inner][col]; K is the reduction dimension.",
    "measurement": "speedup=old/new; GB/s=bytes/(ms*1e6); optimize only the stated fraction.",
    "triton_add": "Use program_id * BLOCK + arange, mask BOTH loads and the store, cdiv grid.",
    "triton_sum": "One program per row; zero-pad loads, convert to float32, sum axis=0.",
    "triton_softmax": "Pad with -inf, reduce max, exponentiate shifted values, reduce sum.",
}


def equal(actual, expected):
    if actual != expected:
        raise AssertionError(f"expected {expected!r}, got {actual!r}")


def close(actual, expected, *, atol=1e-9, rtol=1e-7, path="result"):
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise AssertionError(f"{path}: expected {len(expected)} entries, got {actual!r}")
        for i, (a, b) in enumerate(zip(actual, expected)):
            close(a, b, atol=atol, rtol=rtol, path=f"{path}[{i}]")
    elif not isinstance(actual, (int, float)) or not math.isclose(
        actual, expected, abs_tol=atol, rel_tol=rtol
    ):
        raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")


def unchanged(function, *args):
    original = copy.deepcopy(args)
    function(*args)
    equal(args, original)


def normalized_softmax(module):
    out = module.stable_softmax([-20, 3, 4, -8, 0])
    close(sum(out), 1)
    if not all(0 <= value <= 1 for value in out):
        raise AssertionError(f"Probabilities must be in [0,1], got {out!r}")


def cpu_cases(name):
    """Each returned check is named so a failure teaches a specific property."""
    return {
        "offsets": [
            ("partial final block", lambda m: equal(m.ceil_div(10, 4), 3)),
            ("exact multiple", lambda m: equal(m.ceil_div(12, 4), 3)),
            ("empty vector", lambda m: equal(m.ceil_div(0, 4), 0)),
            ("one element", lambda m: equal(m.ceil_div(1, 256), 1)),
            ("first block", lambda m: equal(m.block_offsets(10, 4, 0), ([0, 1, 2, 3], [True] * 4))),
            (
                "tail mask",
                lambda m: equal(
                    m.block_offsets(10, 4, 2), ([8, 9, 10, 11], [True, True, False, False])
                ),
            ),
        ],
        "strides": [
            ("row gap", lambda m: equal(m.offset(1, 2, 5), 7)),
            ("column stride and base", lambda m: equal(m.offset(1, 2, 10, 2, 3), 17)),
            (
                "cropped rows",
                lambda m: equal(m.read_view(list(range(10)), 2, 3, 5), [[0, 1, 2], [5, 6, 7]]),
            ),
            (
                "transpose",
                lambda m: equal(m.read_view(list(range(6)), 3, 2, 1, 3), [[0, 3], [1, 4], [2, 5]]),
            ),
            (
                "offset slice",
                lambda m: equal(m.read_view(list(range(12)), 2, 2, 4, 1, 1), [[1, 2], [5, 6]]),
            ),
        ],
        "reduction": [
            ("width three", lambda m: equal(m.sum_stages([1, 2, 3]), [[1, 2, 3, 0], [3, 3], [6]])),
            (
                "width five",
                lambda m: equal(
                    m.sum_stages([1, 2, 3, 4, 5]),
                    [[1, 2, 3, 4, 5, 0, 0, 0], [3, 7, 5, 0], [10, 5], [15]],
                ),
            ),
            ("one value", lambda m: equal(m.sum_stages([7]), [[7]])),
            ("negative values", lambda m: equal(m.sum_stages([-1, -2, 3, -4])[-1], [-4])),
            ("input unchanged", lambda m: unchanged(m.sum_stages, [1, 2, 3])),
        ],
        "softmax": [
            (
                "worked row",
                lambda m: close(
                    m.stable_softmax([1, 2, 3]),
                    [0.09003057317038046, 0.24472847105479764, 0.6652409557748218],
                ),
            ),
            ("large positive logits", lambda m: close(m.stable_softmax([1000, 1000]), [0.5, 0.5])),
            (
                "large negative logits",
                lambda m: close(m.stable_softmax([-1000, -1000]), [0.5, 0.5]),
            ),
            ("single element", lambda m: close(m.stable_softmax([123]), [1])),
            (
                "shift invariant",
                lambda m: close(m.stable_softmax([1, 2, 3]), m.stable_softmax([1001, 1002, 1003])),
            ),
            ("normalized probabilities", normalized_softmax),
            ("input unchanged", lambda m: unchanged(m.stable_softmax, [1, 2, 3])),
        ],
        "rmsnorm": [
            (
                "worked row",
                lambda m: close(
                    m.residual_rmsnorm([1, 2], [2, 2], [1, 2], 0.5),
                    [3 / math.sqrt(13), 8 / math.sqrt(13)],
                ),
            ),
            (
                "zero residual sum",
                lambda m: close(m.residual_rmsnorm([1, -2], [-1, 2], [7, 3], 1e-5), [0, 0]),
            ),
            (
                "constant row stays nonzero",
                lambda m: close(m.residual_rmsnorm([2, 2], [0, 0], [1, 1], 5), [2 / 3, 2 / 3]),
            ),
            (
                "weight is per column",
                lambda m: close(m.residual_rmsnorm([1, 1], [0, 0], [2, -3], 3), [1, -1.5]),
            ),
            (
                "input unchanged",
                lambda m: unchanged(m.residual_rmsnorm, [1, 2], [2, 2], [1, 2], 0.5),
            ),
        ],
        "matmul": [
            (
                "rectangular product",
                lambda m: close(
                    m.matmul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]),
                    [[58, 64], [139, 154]],
                ),
            ),
            ("signed dot product", lambda m: close(m.matmul([[2, -3]], [[4], [5]]), [[-7]])),
            (
                "identity",
                lambda m: close(m.matmul([[1, 2], [3, 4]], [[1, 0], [0, 1]]), [[1, 2], [3, 4]]),
            ),
            ("ragged tiles", lambda m: equal(m.tile_ranges(7, 3), [(0, 3), (3, 6), (6, 7)])),
            ("empty tile dimension", lambda m: equal(m.tile_ranges(0, 4), [])),
            ("input unchanged", lambda m: unchanged(m.matmul, [[1, 2]], [[3], [4]])),
        ],
        "measurement": [
            ("two times faster", lambda m: close(m.speedup(10, 5), 2)),
            ("slower candidate", lambda m: close(m.speedup(5, 10), 0.5)),
            ("milliseconds conversion", lambda m: close(m.effective_gbps(1_000_000, 1), 1)),
            ("vector traffic", lambda m: close(m.effective_gbps(12_000_000, 2), 6)),
            ("Amdahl limit", lambda m: close(m.overall_speedup(0.1, 2), 1 / 0.95)),
            ("unchanged program", lambda m: close(m.overall_speedup(0, 100), 1)),
        ],
    }[name]
