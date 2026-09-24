# A1. Merge softmax statistics without losing numerical stability

**Prerequisite:** stable softmax. **Time:** 2-3 sessions, CPU only.
**Edit:** [online_normalizer.py](../exercises/online_normalizer.py).
**Goal:** summarize independent chunks and merge them without seeing every value at once.

For a chunk X, store two numbers:

```text
m = max(X)
l = sum(exp(x-m) for x in X)
```

They represent an unshifted sum `exp(m)*l`, but do not actually calculate exp(m).
That expression can overflow even when the final probabilities are well defined.

## Derive the merge

For chunks A and B, choose the larger maximum `m=max(mA,mB)`. Re-express both
denominators relative to m:

```text
l = lA * exp(mA-m) + lB * exp(mB-m)
merged state = (m,l)
```

Example: A=`[2,2]` gives `(2,2)`; B=`[4]` gives `(4,1)`. Merging gives
`(4, 2*exp(-2)+1)`, approximately `(4,1.270670566)`. The three probabilities are
approximately `[0.106506979,0.106506979,0.786986042]`. Simply adding `2+1` is wrong:
the two sums were computed relative to different maxima.

## Empty chunks and merge trees

Use `(-inf,0)` as an empty identity. Handle either empty side explicitly before
computing exponentials. Merging two empty states naively computes `-inf-(-inf)`,
which is NaN. `summarize([])` returns the identity and `streaming_softmax([],k)`
returns `[]` for a valid positive integer k.

In real arithmetic this merge is associative and commutative. Floating-point
rounding means different merge trees can differ slightly, so check closeness,
not bitwise equality. Python floats are typically FP64; this exercise does not
simulate the FP32 reduction order of a GPU kernel.

## Implement and inspect

1. Implement `summarize(values)` for finite values.
2. Implement `merge(left,right)`, including both identity cases.
3. Implement `streaming_softmax(values,chunk_size)`: summarize chunks, merge their
   states, then reread all values to produce `exp(x-m)/l`.
4. Reject zero, negative and noninteger chunk sizes.

```bash
python -S -m learning.workshops.check online_normalizer
python -S -m learning.workshops.check online_normalizer --solution
```

The checks use logits near +/-1000, partitions of different sizes and merge trees
with very different maxima. Write out the state after each chunk yourself.

## What this does and does not save

You can stream the **statistics** in one pass with constant scalar state. You
cannot emit final softmax probabilities for early values until the final
denominator is known, so this implementation rereads values. Attention changes
the output: it needs a weighted sum of V, allowing that sum to be maintained in
the same recurrence instead of writing every probability.

**Done when:** derive the rescaling equation, handle two empty states, explain why
the output needs a second pass, and describe why merging normalized probabilities
alone loses the information needed to combine chunks.

Reference: [Milakov and Gimelshein, Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867).
Next: [split softmax on the GPU](05_split_softmax.md).
