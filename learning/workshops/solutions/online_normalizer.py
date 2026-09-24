"""A1: mergeable normalization state using Python floats, not a GPU simulator."""

import math

EMPTY = (-math.inf, 0.0)


def summarize(values):
    if not values:
        return EMPTY
    maximum = max(values)
    return maximum, math.fsum(math.exp(x - maximum) for x in values)


def merge(left, right):
    # Explicit identity handling avoids exp(-inf - -inf).
    if left[1] == 0:
        return right
    if right[1] == 0:
        return left
    maximum = max(left[0], right[0])
    total = left[1] * math.exp(left[0] - maximum) + right[1] * math.exp(right[0] - maximum)
    return maximum, total


def streaming_softmax(values, chunk_size):
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    state = EMPTY
    for start in range(0, len(values), chunk_size):
        state = merge(state, summarize(values[start : start + chunk_size]))
    maximum, total = state
    return [math.exp(x - maximum) / total for x in values]
