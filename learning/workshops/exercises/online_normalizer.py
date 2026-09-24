"""A1: finite Python floats only; see ../lessons/04_online_normalizer.md."""


def summarize(values):
    """Return (max, sum(exp(x-max))); empty input returns (-inf, 0)."""
    raise NotImplementedError("Compute a numerically stable chunk summary")


def merge(left, right):
    """Merge two summaries, treating (-inf, 0) as identity on either side."""
    raise NotImplementedError("Rescale both sums to a shared maximum")


def streaming_softmax(values, chunk_size):
    """Merge summaries, then reread values to normalize. Validate chunk_size."""
    raise NotImplementedError("Use summarize and merge; return [] for empty input")
