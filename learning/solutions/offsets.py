"""Reference answer: read after attempting learning/exercises/offsets.py."""


def ceil_div(n, block_size):
    return (n + block_size - 1) // block_size


def block_offsets(n, block_size, program_id):
    offsets = [program_id * block_size + lane for lane in range(block_size)]
    return offsets, [index < n for index in offsets]
