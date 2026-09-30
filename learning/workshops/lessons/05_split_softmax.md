# A2. Split a wide softmax row across programs

**Prerequisite:** A1 and a working Triton row reduction. **Time:** 3-4 sessions.
**Edit:** [split_softmax.py](../exercises/split_softmax.py).
**Goal:** support rows through 131072 columns without one enormous per-row block.

The portfolio's `_softmax_looped` handles wide rows with one program per row that
loops over chunks. That works well with many rows, but with only a few rows it
leaves most of the GPU idle; see the
[case study](../../../docs/CASE_STUDIES.md#wide-rows-loop-in-one-program-or-split-across-programs).
Splitting each row across programs, as here, is the answer for that case.
Increasing that cap alone can increase register pressure and resource use. With
only a few rows it also exposes few independent programs. Partitioning a row
creates more independent work, at the cost of scratch, rereads and launches.

## Three stages with explicit ownership

For M rows, width N and chunk C, define `T=ceil(N/C)`:

```text
X[M,N]
  | partials: grid(M,T), each program owns one chunk
  v
partial_max[M,T], partial_sum[M,T]  (FP32)
  | merge: grid(M), rescale all partial sums to one maximum
  v
global_max[M], global_sum[M]        (FP32)
  | normalize: grid(M,T), reread a chunk and write its probabilities
  v
Y[M,N]
```

For N=1025,C=1024, T=2. Chunk zero has 1024 elements; chunk one has one real
element and 1023 masked lanes. Both chunks contain at least one valid element.
Pad partial loads with `-inf`, reduce a maximum, then compute the shifted sum.

The merge kernel applies the A1 equation to all partial states. Its padded
partial maxima are `-inf` and partial sums are zero. The normalize kernel reads
the final maximum and denominator and writes only its own output chunk.

## Synchronization and memory

Launch all stages on the current CUDA stream. Stream ordering makes earlier
kernel writes available to later kernels. A block barrier inside the first kernel
does not synchronize independent programs, and a hand-written global spin barrier
can deadlock if not all blocks can reside concurrently.

Allocate scratch for each call. Shared global scratch could be overwritten by a
second call on another stream. Do not synchronize the entire device inside the
operator; that would change composition and benchmark behavior.

FP32 scratch occupies `8*M*(T+1)` bytes: two partial arrays plus two global arrays.
For M=4,N=32769,C=1024, T=33 and scratch is 1088 bytes. This is an allocation-size
calculation, not a measurement of physical memory traffic. The input is read twice
and the output written once, before accounting for scratch/cache/spills.

## Implement and test

1. Implement partial statistics and compare them by hand on `[2,2]` and `[4]`.
2. Implement the rescaled merge. Adding raw partial denominators is incorrect.
3. Normalize using the merged state. Mask stores and keep inputs unchanged.
4. Validate contiguous input, supported width and C in `{256,1024,4096}`.

```bash
python -m learning.workshops.check split_softmax
python -m learning.workshops.benchmark --op split_softmax --output results/local/a2.json
```

The checker covers the width cap, ragged final chunks, large constant rows,
isolated dominant logits, three dtypes, all chunk options and a call made on a
non-default stream. That last check compares results after synchronizing, so it
cannot prove every stage launched on the caller's stream; review your launches
for that.

## Form a performance hypothesis

Smaller C exposes more programs but increases partial count and merge work.
Larger C reduces partials but asks each program to reduce more values. Compare
4x32769 and 64x131072 against `torch.softmax`. Three launches can lose badly on
small workloads; a valid improvement may be support for a shape, not a speedup.

The main implementation now uses this idea for softmax and RMSNorm when a GPU has
more SMs than rows, with one change: there is no separate merge kernel, because
every normalizing program merges its row's partial statistics itself. After your
attempt, compare with `_softmax_split_stats` and `_softmax_split_normalize` in
[triton_kernels.py](../../../src/kernel_portfolio/triton_kernels.py); the
[case study](../../../docs/CASE_STUDIES.md#wide-rows-loop-in-one-program-or-split-across-programs)
measures when it pays.

**Done when:** explain the launch boundaries, derive the scratch size, show why
the tail chunk is nonempty, and retain the losing chunk choices in your report.
Next: [stream the attention numerator too](06_streaming_attention.md).
