"""Reference answer: keep named intermediates so the equation is visible."""

import math


def stable_softmax(values):
    maximum = max(values)
    numerator = [math.exp(value - maximum) for value in values]
    denominator = sum(numerator)
    return [value / denominator for value in numerator]
