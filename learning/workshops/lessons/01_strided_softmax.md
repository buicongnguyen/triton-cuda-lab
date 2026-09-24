# I1. Softmax over real tensor views

**Prerequisite:** beginner lessons 2, 4 and 8. **Time:** 2-3 sessions.
**Edit:** [strided_softmax.py](../exercises/strided_softmax.py).
**Goal:** handle a view without assuming its logical neighbors are adjacent in memory.

## Work out the addresses

Consider contiguous storage `[0,1,2,3,4,5]`, viewed as a 2x3 matrix. Transpose it
to obtain `[[0,3],[1,4],[2,5]]`. The new shape is 3x2 and its strides are `(1,3)`.

| Logical row | Logical column | `row*1 + col*3` | Value |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 0 |
| 0 | 1 | 3 | 3 |
| 2 | 0 | 2 | 2 |
| 2 | 1 | 5 | 5 |

The pointer passed to Triton already addresses the first element of the view.
Do not add PyTorch's storage offset a second time. A slice such as
`base[1:, 1::2]` combines a shifted pointer with a column stride of two.

The old portfolio softmax intentionally rejects noncontiguous columns. In this
workshop the load is `X + row*SR + col*SC`. The output has a newly allocated
contiguous layout, so its store is `Y + row*N + col`. Input and output addressing
are different even though the tensors have identical shapes.

## Implement in three passes

1. Add a `@triton.jit` kernel with one program per logical row. Pass both input
   strides and width. Use the next power of two for the reduction block.
2. Add stable FP32 softmax, negative-infinity tail padding, and masked stores.
3. Validate with `learning.workshops.contracts.matrix` and `warps`, allocate the
   output, and launch under the input's CUDA device context. Empty row counts
   return the empty output without launching.

Read-only broadcast inputs with stride zero are valid: several logical entries
can read one address. Output entries must still have distinct addresses. This
does not authorize overlapping writes to the input.

```bash
python -m learning.workshops.check strided_softmax --hint
python -m learning.workshops.check strided_softmax
python -m learning.workshops.benchmark --op strided_softmax --output results/local/i1.json
```

## Experiment, not just correctness

The benchmark constructs transposed inputs. Compare direct loads using 4 and 8
warps with a variant that calls `x.contiguous()` before the same kernel. The copy
belongs inside the timed function. Predict whether the extra traffic can pay for
better access during reduction, then inspect both shapes and both timing modes.

Logical strides explain requested addresses, not actual DRAM transactions. A
claim about coalescing efficiency needs generated-code/profiler evidence. Start
with the address pattern and a measured runtime; do not invent hardware counters.

**Debugging clues:** a contiguous case passes but transpose fails -> check SC;
only slices fail -> check double-counted offset; values are right but shape/layout
is wrong -> check output allocation; probabilities are too small -> inspect padding.

**Done when:** all checks pass, you draw the transpose addresses without code,
and your worksheet reports the cost of the copy rather than excluding it.

Reference: [Triton softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html).
Next: [derive the backward pass](02_softmax_backward.md).
