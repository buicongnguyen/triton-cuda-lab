"""Lesson 5. RMSNorm scales a row without subtracting its mean."""

import math


def residual_rmsnorm(x, residual, weight, eps):
    """Return z / sqrt(mean(z*z) + eps) * weight where z = x + residual.

    Inputs are equal-length nonempty lists; eps is positive. Do not modify them.
    Python floats provide the conceptual math, not an FP16 simulation.
    """
    _ = math
    raise NotImplementedError("Build z, compute one row scale, then scale each element")
