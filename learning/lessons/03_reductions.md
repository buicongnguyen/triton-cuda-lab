# 3. Reductions: many values become one

**Time:** 60 minutes. **Prerequisite:** offsets and strides.
**Edit:** [reduction.py](../exercises/reduction.py).

An elementwise add reads two values and writes one value at the same position.
A reduction combines an entire group into fewer values: sum, maximum and mean
are examples. Values must cooperate because one output depends on many inputs.

## Trace the arithmetic

Start with `[1,2,3,4,5]`. Pad to eight elements, then combine neighbors:

```text
level 0:  [1, 2, 3, 4, 5, 0, 0, 0]
level 1:  [  3,    7,    5,    0  ]
level 2:  [      10,          5     ]
level 3:  [             15          ]
```

Eight values need three pairwise levels (`log2(8)`). Several pairs at a level can
be computed simultaneously. This is a conceptual tree: Triton's actual GPU tree
depends on layout, dtype and compiler choices.

## Padding has an algebraic meaning

An **identity** leaves the operation unchanged. Adding zero does not change a
sum. Taking `max(x, -infinity)` leaves any finite x unchanged.

| Reduction | Identity for padding | Why the wrong value fails |
| --- | --- | --- |
| sum | 0 | Padding with 1 changes the sum |
| maximum | negative infinity | Padding negative values with 0 invents a larger maximum |
| minimum | positive infinity | Padding positive values with 0 invents a smaller minimum |
| product | 1 | Padding with 0 makes the entire product zero |

For the **mean**, sum padded values but divide by the logical count. The mean of
`[1,2,3,4,5]` is 15/5=3, not 15/8. This detail will matter again in RMSNorm.

## What FP32 accumulation means

A dtype limits which values are representable. In FP16, converting intermediate
partial sums back to FP16 can lose information or overflow. The row-sum kernel
loads the input, converts values to FP32, reduces in FP32 and returns FP32.
Input storage can remain FP16; storage dtype and computation dtype need not match.

Even FP32 is not exact. In FP32 arithmetic, adding 1 to 100,000,000 rounds back
to 100,000,000. Then subtracting 100,000,000 yields zero. Reordering the first
addition to `100,000,000 + (-100,000,000)` yields zero, after which adding 1 yields
one. Different reduction orders can therefore produce different answers.
Python's ordinary float has more precision, so your Python exercise is about the
structure of the tree, not an exact simulation of FP32 rounding.

## Implement the exercise

Return every stage, including the padded input and the final one-element list.
Copy the input before padding so the caller's list stays unchanged. A one-element
input already has its final stage and needs no pairwise step.

```bash
python -m learning.explain reduction
python -m learning.check reduction
```

## Translate the idea to GPU code

In `_row_sum`, `tl.program_id(0)` chooses a row and `tl.arange` chooses its columns.
`tl.sum(values, axis=0)` reduces the one-dimensional vector of loaded values into
a scalar. The output pointer is `OUT + row`, not `OUT + row * width`, because
there is only one output per row.

The CUDA example makes the cooperation explicit: warp shuffles combine thread
partials; eight warp leaders write into shared memory; the block synchronizes;
one warp combines the partials and broadcasts the result. Read that code after
completing the Triton version, rather than memorizing shuffle syntax first.

**Exit check:** draw all stages for `[2,-1,4]`. State the sum identity, max identity,
and correct divisor for a padded mean. Next: [softmax](04_softmax.md).
