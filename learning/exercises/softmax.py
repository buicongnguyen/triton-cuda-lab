"""Lesson 4. Python lists stand in for one GPU row."""

import math


def stable_softmax(values):
    """Return probabilities for a nonempty list of finite numbers.

    Steps: largest value -> subtract -> math.exp -> sum -> divide.
    For [1000, 1000], return [0.5, 0.5] without overflowing.
    """
    # Keep math imported for your implementation.
    _ = math
    raise NotImplementedError("Subtract the row maximum before exponentiating")
