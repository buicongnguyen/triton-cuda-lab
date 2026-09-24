# Turn a workshop into evidence

Choose one workshop at a time. Predict a result, implement it, check it, measure
it, and explain the losing case. Keep this template blank until you do the work.

## Experiment record

Copy this section into a new personal note:

```text
Question and workload:
Input shape / strides / dtype / layout:
GPU / driver / PyTorch / Triton:
Correctness contract and tolerance:
Reference algorithm and independent check:
Hypothesis (before timing):
One implementation variable changed:
Baseline expression and its rounding semantics:
Command / source hash / raw JSON location:
Timing mode / samples / iterations / warmup:
Median latency and min/max for each variant:
Speedup denominator (which baseline):
What won, what lost, and which claim the evidence supports:
What cannot be concluded without additional evidence:
Next experiment:
```

Do not report the best single sample as typical performance. Compare medians and
dispersion, preserve raw samples, and rerun if the difference is within observed
noise. Do not combine measurements from different timing modes into one ratio.

## Six practice sessions after implementation

| Session | Change one thing | Required comparison | Explain aloud |
| --- | --- | --- | --- |
| 1 | I1 direct access vs copy+kernel | Transposed 256x257 and 1024x1024, copy included | Why a copy might pay for itself, and when it does not |
| 2 | I2 fused VJP vs eager composition | 256x1025 and 1024x4097 | Why an N-by-N Jacobian is unnecessary |
| 3 | I3 GROUP and then BM | Tiny ragged and 512-cubed GEMM | Why the FP32 epilogue baseline matters |
| 4 | A1 chunk partition and merge tree | +/-1000 logits; empty states | Why the maximum is part of the state |
| 5 | A2 chunk size | Few wide rows vs many wide rows | Parallelism versus extra launches and scratch |
| 6 | A3 BN=32/64 | Dense eager and PyTorch SDPA, causal=True | What online numerator rescaling saves |

## Optional capstones (not already implemented)

Pick one after all relevant checks pass. These are bounded follow-on tasks, not
features claimed by the current reference implementations.

**Capstone A: skip future attention tiles.** Bound the causal key loop by the
largest real query index in a query tile. Keep the mask within the diagonal tile.
Test N=1/17/33/65/129 and both BN values, compare every output to the original
oracle, and rerun the future-V perturbation test. Measure both small and large N.
Done means preserved correctness plus reported graph/event measurements, even
if performance does not improve. Do not add arbitrary masks in the same change.

**Capstone B: add a reusable workspace API for split softmax.** Define scratch
shape/device/dtype and ownership explicitly. Require one workspace per concurrent
call or protect ownership at the caller. Test two CUDA streams with distinct
workspaces; demonstrate why sharing unsynchronized scratch is invalid. Compare
allocation-inclusive calls and caller-preallocated workspace calls separately.
Do not subtract allocator time from one side only.

**Capstone C: implement the wide softmax stages in CUDA C++.** Start from the
existing [CUDA example](../../cuda/kernels.cu), but give each stage its own kernel.
Map rows/chunks to blocks, use a block reduction for partial stats, and launch
all stages on a passed stream. Check the same ragged widths and FP32 reference;
run Compute Sanitizer memcheck and racecheck before profiling. Compare the
three-stage algorithm with Triton using equivalent timing. This is an explicit
CUDA extension exercise; the repository does not yet contain that port.

## A five-minute demonstration

1. State the operator contract and draw ownership of one output tile.
2. Derive one formula or index mapping with the lesson's tiny example.
3. Show an independent correctness check, including a failure your check catches.
4. Show raw measurement evidence against a clearly named baseline.
5. Explain a limitation and the next bounded experiment.

An honest result such as “the stream recurrence avoids a dense score allocation,
but this small implementation loses to SDPA on my tested GPU” is stronger than
a universal speedup claim unsupported by the experiment.
