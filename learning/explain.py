"""Print tiny execution traces with python -m learning.explain TOPIC.

Topics: indexing, strides, reduction, softmax, matmul.
"""

import argparse
import math


def indexing():
    n, block = 10, 4
    print("N=10, BLOCK=4: ceil(10/4)=3 programs. Mask before accessing memory.")
    print("program  local  offset  valid  action")
    for program in range(3):
        for lane in range(block):
            offset = program * block + lane
            print(
                f"{program:7} {lane:6} {offset:7} {str(offset < n):6} "
                f"{'load/store' if offset < n else 'masked out'}"
            )


def strides():
    storage = list(range(10))
    print(f"Storage: {storage}\nLogical view: 2 rows x 3 columns; row stride=5.")
    for row in range(2):
        for col in range(3):
            offset = row * 5 + col
            print(f"view[{row},{col}] -> {row}*5+{col} = storage[{offset}] = {storage[offset]}")
    print("Using row*3+col would read the wrong second row: [3,4,5] instead of [5,6,7].")


def reduction():
    current = [1, 2, 3, 4, 5, 0, 0, 0]
    print("Input [1,2,3,4,5] padded with sum's identity, zero:")
    print(current)
    while len(current) > 1:
        next_values = []
        for index in range(0, len(current), 2):
            value = current[index] + current[index + 1]
            print(f"  {current[index]} + {current[index + 1]} = {value}")
            next_values.append(value)
        current = next_values
        print(current)
    print("This illustrates a reduction; Triton can choose a different hardware tree.")


def softmax():
    values = [1.0, 2.0, 3.0]
    maximum = max(values)
    shifted = [v - maximum for v in values]
    numerator = [math.exp(v) for v in shifted]
    denominator = sum(numerator)
    print(f"input        = {values}\nmaximum      = {maximum}\nshifted      = {shifted}")
    print(f"exp(shifted) = {[round(v, 6) for v in numerator]}\nsum          = {denominator:.6f}")
    print(f"probabilities= {[round(v / denominator, 6) for v in numerator]}")
    print("Add 1000 to every input: shifted values and probabilities stay the same.")
    print("An extra -inf padding lane contributes exp(-inf)=0 to the denominator.")


def matmul():
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[7, 8], [9, 10], [11, 12]]
    print(f"A[2,3]={a}\nB[3,2]={b}\nC[2,2]=A@B; K=3, toy BK=2.")
    for row in range(2):
        for col in range(2):
            total = 0
            for start in (0, 2):
                terms = [f"{a[row][k]}*{b[k][col]}" for k in range(start, min(start + 2, 3))]
                subtotal = sum(a[row][k] * b[k][col] for k in range(start, min(start + 2, 3)))
                total += subtotal
                print(
                    f"C[{row},{col}], K tile {start // 2}: {' + '.join(terms)} = {subtotal}; "
                    f"accumulator={total}"
                )
    print("BK=2 is a paper example, not a valid Tensor Core tile configuration here.")


def main():
    demos = {
        "indexing": indexing,
        "strides": strides,
        "reduction": reduction,
        "softmax": softmax,
        "matmul": matmul,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topic", choices=demos)
    args = parser.parse_args()
    demos[args.topic]()


if __name__ == "__main__":
    main()
