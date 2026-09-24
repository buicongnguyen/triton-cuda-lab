"""Reference answer: scalar loop structure underlying the tiled GPU kernel."""


def matmul(a, b):
    m, k, n = len(a), len(b), len(b[0])
    out = [[0.0 for _ in range(n)] for _ in range(m)]
    for row in range(m):
        for col in range(n):
            for inner in range(k):
                out[row][col] += a[row][inner] * b[inner][col]
    return out


def tile_ranges(length, tile_size):
    return [(start, min(start + tile_size, length)) for start in range(0, length, tile_size)]
