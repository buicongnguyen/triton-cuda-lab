# 4. Softmax: derive the kernel before optimizing it

**Time:** 75–90 minutes. **Prerequisite:** reductions and Python `math.exp`.
**Edit:** [softmax.py](../exercises/softmax.py).

Softmax turns a row of scores into nonnegative values that sum to one:

```text
y[i] = exp(x[i]) / sum_j(exp(x[j]))
```

The whole row matters. Changing one score changes the denominator and therefore
can change every output. This is why independently applying exp to each element
is not a complete softmax implementation.

## Calculate [1,2,3] by hand

First subtract the largest input, 3. The probabilities are unchanged because
every numerator and the denominator have the same factor `exp(-3)`.

| Column | Input x | x - max(x) | Exponential | Divide by 1.503214724 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1 | -2 | 0.135335283 | 0.090030573 |
| 1 | 2 | -1 | 0.367879441 | 0.244728471 |
| 2 | 3 | 0 | 1.000000000 | 0.665240956 |

Check that the last column sums to one. If the inputs become `[1001,1002,1003]`,
the shifted inputs are still `[-2,-1,0]`, so the output is unchanged. Directly
evaluating `math.exp(1003)` would overflow; the stabilized formula avoids that.

## Six named steps

1. `maximum = max(values)` — a reduction.
2. `shifted[i] = values[i] - maximum` — elementwise work.
3. `numerator[i] = exp(shifted[i])` — elementwise work.
4. `denominator = sum(numerator)` — a second reduction.
5. `output[i] = numerator[i] / denominator` — elementwise work.
6. Store the valid outputs.

Write these as named Python variables before compressing anything into one line.

```bash
python -m learning.explain softmax
python -m learning.check softmax
```

The checker deliberately uses both `[1000,1000]` and `[-1000,-1000]`, a single
element, a shifted row, and a probability-sum check. Each case has a purpose.

## Why padding must be negative infinity

For width 3, a four-element Triton block has one extra lane. Loading
`[1,2,3,-inf]` preserves the maximum, and `exp(-inf-3)=0` preserves the sum.
Zero padding is wrong: it introduces a positive exponential into the denominator.
Even if zero does not change a positive maximum, the probabilities still change.
For a negative row such as `[-3,-2,-1]`, it changes the maximum too.

Masking only the final store does not fix an incorrect denominator: the padded
lane already influenced every valid output during the reduction.

## What fusion saves, with explicit assumptions

Imagine each step produces a full array in GPU global memory. The shifted array
and exponential array have to be written and then read by later steps. A fused
kernel loads a row once, computes its reductions and intermediate values within
the program, then stores the output once.

For a 1024-by-1024 FP32 tensor, one input read plus one output write is
`2 * 1024 * 1024 * 4 = 8,388,608` bytes, or 8 MiB of logical traffic. That is a
lower-bound model, not proof of the actual DRAM bytes. Caches, compiler spills,
layout and repeated data reuse affect hardware traffic.

The repo compares against **both** a readable eager decomposition and
`torch.softmax`. The latter is already specialized. A large win over an intentionally
unfused expression does not guarantee a win over the built-in operation.

## Follow the actual implementation

Read `_softmax` in [triton_kernels.py](../../src/kernel_portfolio/triton_kernels.py).
Point to the two reductions, the conversion to FP32, the padded load value and
the output store mask. Up to width 8192 a row fits one block; wider rows (up to
2^20) use `_softmax_looped`, which keeps an online maximum and sum per lane, the
idea workshop A1 develops. The wrapper rejects noncontiguous columns. The examples
assume finite inputs: an all-negative-infinity row needs a separately defined
policy because subtracting its maximum produces NaNs.

**Exit check:** explain why max subtraction preserves softmax, why exp alone is
insufficient, and why zero padding is wrong even for a positive row. Then move
to [RMSNorm](05_rmsnorm.md). You will implement the GPU version in lesson 8.

Further reading after the exercise: [official Triton fused-softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html).
