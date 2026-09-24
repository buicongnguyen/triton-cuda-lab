"""Reference answer: a teaching tree, not a claim about exact compiler scheduling."""


def sum_stages(values):
    block_size = 1 << (len(values) - 1).bit_length()
    current = list(values) + [0] * (block_size - len(values))
    stages = [current]
    while len(current) > 1:
        current = [current[i] + current[i + 1] for i in range(0, len(current), 2)]
        stages.append(current)
    return stages
