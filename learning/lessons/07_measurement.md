# 7. A timing number needs an explanation

**Time:** 60 minutes. **Prerequisite:** the previous operations and basic ratios.
**Edit:** [measurement.py](../exercises/measurement.py).

## Three quantities you should calculate yourself

| Quantity | Formula | Worked example |
| --- | --- | --- |
| Speedup | baseline time / candidate time | 10 ms / 5 ms = 2x |
| Effective bandwidth | logical bytes / (elapsed_ms * 1e6) | 12,000,000 bytes / 2,000,000 = 6 GB/s |
| Overall speedup | 1 / ((1-f) + f/s) | f=0.1, s=2 gives 1/0.95 ≈ 1.053x |

The last row is Amdahl's law. If a kernel consumes only 10% of a program, doubling
that kernel's speed saves 5% of original total time. It cannot double the whole program.

For FP32 vector add, logical traffic is two input reads plus one output write,
or `3*N*4` bytes. For fused FP32 softmax, the ideal useful traffic is `2*M*N*4`
bytes. “Effective” means it is derived from this model. Only profiler counters
could establish actual DRAM traffic; this machine currently lacks those permissions.

```bash
python -m learning.check measurement
```

## Why a Python stopwatch can mislead you

A GPU launch is asynchronous. Timing only `fn()` can stop before the device has
finished. First execution can also include kernel compilation, tuning and allocator
setup. Those are useful costs to measure separately, but not steady-state kernel time.

The repo's default benchmark warms the callable, captures repeated invocations in
a CUDA graph, times replay with CUDA events and divides by the invocation count.
That estimates a repeated device workload. `--timing events` submits calls through
Python and can include gaps while the host prepares the next launch.

## Read two actual measurements

For FP16 softmax with shape 1024x1024, the saved runs show:

| Timing mode | PyTorch | Triton, 4 warps | baseline/candidate |
| --- | ---: | ---: | ---: |
| Graph replay, inputs warm in L2 | 4.198 us | 2.389 us | 1.76x |
| Graph replay, L2 flushed before each call | 12.186 us | 9.933 us | 1.23x |
| Ordinary event-timed wrapper | 11.366 us | 33.835 us | 0.34x |

These rows answer different questions. The kernel's captured device work can be
fast while the validated Python wrapper is expensive for tiny calls. The 2 MiB FP16
input and 2 MiB output fit in this GPU's 64 MiB L2, so the warm row measures cache-resident repeats; the
flushed row reads from DRAM and shows a smaller advantage. On this desktop GPU,
shared with other applications, the flushed ratio for this shape ranged from 1.23x
to 1.67x across runs, a reminder to repeat a measurement before trusting a digit.
The measurements do not justify calling every use of the custom operator faster. See the
[raw evidence and explanation](../../docs/CASE_STUDIES.md).

## Your first measured experiment

**GPU step.** The exercise above is CPU-only. These commands need the GPU
environment, so run them after lesson 8 (the [schedule](../SCHEDULE.md) does this
in week 5). Each prints one line per variant and writes a JSON file.

```bash
kernel-bench --op softmax --shape 1024 1024 --dtype float32 --compile --output results/local/my-softmax-graph.json
kernel-bench --op softmax --shape 1024 1024 --dtype float32 --timing events --output results/local/my-softmax-events.json
# Optional: flush L2 before every call, so inputs come from DRAM
kernel-bench --op softmax --shape 1024 1024 --dtype float32 --cache cold --output results/local/my-softmax-cold.json
```

To read a run as a table instead of raw JSON:

```bash
python scripts/report_results.py results/local/my-softmax-graph.json --output results/local/my-softmax.md
```

Then open a JSON file and locate `environment`, `settings`, `shape`, `dtype`,
`median_ms`, `samples_ms` and `max_abs_error`. Multiply milliseconds by 1000
to convert to microseconds. Do not compare a FP16 graph result to a FP32 event
result and attribute the difference to the kernel alone.

Write a prediction before running: “I expect four warps to beat eight for this
shape.” Afterward write the observed medians even if the prediction was wrong.
One failed hypothesis that you can explain is stronger evidence than a copied chart.

## A defensible result sentence

> On [GPU], for [shape and dtype], [variant] took [median] under [timing mode],
> versus [baseline and median]. It passed [correctness check]. The result does
> not establish [specific remaining question].

Do not infer low DRAM bandwidth from a timing number alone, or claim a 3% win
with unstable samples. Read min/max and rerun only when noise or a changed
hypothesis justifies it. Next: [your first Triton kernels](08_triton.md).
