"""Reference answer: one inverse RMS per row, one learned weight per column."""

import math


def residual_rmsnorm(x, residual, weight, eps):
    z = [a + b for a, b in zip(x, residual)]
    mean_square = sum(value * value for value in z) / len(z)
    inverse_rms = 1 / math.sqrt(mean_square + eps)
    return [value * inverse_rms * scale for value, scale in zip(z, weight)]
