# A3. Tiled attention with online normalization

**Prerequisite:** I3 and A1; A2 is recommended. **Time:** 4-5 sessions.
**Edit:** [streaming_attention.py](../exercises/streaming_attention.py).
**Goal:** compute `softmax(QK^T/sqrt(D)) @ V` without allocating the N-by-N scores.

Start with a bounded problem: one head, Q/K/V all contiguous N-by-D, equal lengths,
FP16/BF16, N<=2048 and D in `{16,32,64}`. No dropout, arbitrary masks, batching or
backward. This exposes the algorithm without hiding layout decisions behind a
large attention framework.

## Keep three quantities for each query row

One program owns BM=16 query rows. It loads Q once and loops over BN=32 or 64 key
rows. Per query it carries a maximum m, denominator l, and D-element numerator a:

```text
m = -inf; l = 0; a = zeros(D)
for each K/V tile:
    s = Q_tile @ K_tile.T / sqrt(D)
    mask key tails and (if causal) keys whose index exceeds the query index
    m_new = max(m, row_max(s))
    alpha = exp(m - m_new)
    p = exp(s - m_new)
    l = alpha*l + row_sum(p)
    a = alpha*a + p @ V_tile
    m = m_new
output = a/l
```

All state is FP32. Dot products load FP16/BF16 inputs; p is converted to the input
dtype before the PV dot product. This conversion is a deliberate performance and
precision tradeoff, so compare against an independent FP64 attention oracle with
dtype-appropriate tolerances.

## Trace one query before writing the kernel

Assume its already-scaled logits arrive as `[0,0]`, then `[log(2)]`. Let scalar V
values be `[1,3]`, then `[10]`. After the first tile `(m,l,a)=(0,2,4)`.
For the second tile m increases to log(2), so alpha=1/2:

```text
l_new = 0.5*2 + 1 = 2
a_new = 0.5*4 + 1*10 = 12
output = 12/2 = 6
```

The full softmax weights are `[1/4,1/4,1/2]`, giving the same 6. If you rescale l
but forget a, the result becomes 7. This is a numerical bug, not an acceptable
approximation or a launch-tuning issue.

## Masks that preserve the recurrence

Causal attention includes the diagonal: `key_index <= query_index`. Every real
query can see at least key zero, so its first processed tile has a finite maximum
for finite logits. Later wholly future tiles have p=0 and leave the accumulated
state unchanged. Padded query rows have masked output stores.

An arbitrary mask that hides every key would need a separately defined output
policy and guarded recurrence. It is outside this workshop's contract. Likewise,
rectangular Q/K lengths are rejected rather than assigned an ambiguous causal offset.

## Implement in four stages

1. Implement noncausal attention for one tile, then multiple K/V tiles.
2. Add online max/sum/numerator rescaling; test logits whose maxima change by tile.
3. Mask ragged N tails and add causal masking. Keep output row ownership disjoint.
4. Expose BN=32/64 and compare register/work tradeoffs with actual timing.

```bash
python -m learning.workshops.check streaming_attention
python -m learning.workshops.benchmark --op streaming_attention --output results/local/a3.json
```

The checker includes single elements, ragged lengths, both supported BN values,
three head dimensions, both dtypes, maximum N, uniform logits and large logits.
It modifies future V rows and verifies earlier causal outputs stay unchanged.

## Make the comparison credible

The primary benchmark is `torch.nn.functional.scaled_dot_product_attention` with
automatic backend selection. A second dense FP32 expression provides an algorithmic
comparison. Winning against dense attention does not imply beating optimized SDPA.
The reference currently visits future causal tiles rather than skipping them.

At N=2048 a single dense FP32 score matrix contains 16 MiB; this kernel never
allocates it. That does not prove a 16 MiB reduction in peak allocated memory:
the complete baseline may allocate other tensors or use a fused backend, and a
compiler may spill kernel state. Label analytic storage and measured peak memory
separately.

**Done when:** reproduce the answer 6 above, explain every carried state, pass
the causal future-perturbation test, and report your result against SDPA, including
a loss if that is what you measure.

Reference: [Triton's fused-attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).
For follow-on work, choose one [capstone](../EXPERIMENTS.md) and define its contract first.
