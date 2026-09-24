# 5. Residual RMSNorm: one scale for a row

**Time:** 60 minutes. **Prerequisite:** sum, mean, square root and broadcasting.
**Edit:** [rmsnorm.py](../exercises/rmsnorm.py).

A residual connection adds another activation to the current activation. RMSNorm
then divides by the root mean square, with a small positive epsilon, and applies
one learned weight per column. Here is the exact operation used in this repo:

```text
z[i]         = float32(x[i]) + float32(residual[i])
mean_square  = sum_i(z[i] * z[i]) / N
inverse_rms  = 1 / sqrt(mean_square + eps)
output[i]    = cast_to_input_dtype(z[i] * inverse_rms * float32(weight[i]))
```

`inverse_rms` is one scalar shared across the row. `weight[i]` varies by column.
Multiplying a vector by a row scalar is a simple example of broadcasting.

## Work a two-element row

Use `x=[1,2]`, `residual=[2,2]`, `weight=[1,2]`, `eps=0.5`.

| Step | Result |
| --- | --- |
| Residual addition | z=[3,4] |
| Squares | [9,16] |
| Mean square | (9+16)/2=12.5 |
| Add epsilon | 13.0 |
| Inverse RMS | 1/sqrt(13) ≈ 0.277350098 |
| Scale and weight | [3/sqrt(13), 8/sqrt(13)] ≈ [0.832050294, 2.218800785] |

The relatively large epsilon makes the worked example easy to verify. The actual
operator defaults to 1e-5. For an all-zero residual sum, positive epsilon prevents
division by zero and the final output remains zero.

## Three easy mistakes

1. **Divide by the padded width.** Only N real values belong in the mean, even
   when a Triton block pads the row to a power of two.
2. **Subtract a mean.** That changes the operation toward LayerNorm. RMSNorm
   uses the mean of squares without first centering the input.
3. **Round the residual sum too early.** Adding two FP16 tensors and saving an
   FP16 intermediate can differ from converting each input to FP32 before adding.

For an exact FP16 example: 2048 and 1 are individually representable. Their sum
2049 is halfway between neighboring FP16 values and rounds to 2048 with the usual
round-to-nearest-even rule. Converting inputs to FP32 before addition retains 2049.
The normalization denominator is then computed from different numbers. This repo
deliberately chooses the FP32-add contract, and its reference follows that choice.

## Implement the CPU exercise

```bash
python -m learning.check rmsnorm
python -m learning.check rmsnorm --hint
```

Use three small lists for residual sums, squares and output. Once the implementation
passes, eliminate an intermediate list on paper. Decide which values you would
want to keep inside a fused GPU program.

## Read the fused kernel and benchmark fairly

Find `_rmsnorm` in [triton_kernels.py](../../src/kernel_portfolio/triton_kernels.py).
Trace `x`, `r`, `w`, `z`, `inv`. The temporary residual sum is not returned as a
tensor. This is useful only if the caller does not need that intermediate output.

**GPU step (optional now).** This command needs the GPU environment from
lesson 8's setup. Skip it on your first pass and come back later.

```bash
kernel-bench --op rmsnorm --compile --output results/local/my-rmsnorm.json
```

Read the measured [case study](../../docs/CASE_STUDIES.md). For the tiny case,
compilation removes most of the eager overhead, leaving little difference from
the manual kernel. The correct claim includes the comparator and shape.

**Exit check:** state which values are per-element, per-row and per-column; derive
the zero-input result; explain why changing residual precision changes the task.
Next: [matrix multiplication](06_matmul.md).
