# Six labs: from first program to a defensible optimization

**Beginner prerequisite:** complete the [worked lessons and editable exercises](../learning/README.md)
first. This page is the next stage: changing configurations and explaining measurements
after you can write a correct kernel. For session-by-session tasks, use the
[18-session schedule](../learning/SCHEDULE.md).

For new implementations beyond the original five operators, use the
[intermediate and advanced workshops](../learning/workshops/README.md): strided
softmax, backward, fused GEMM, split softmax and streaming attention.

Use the setup in [SETUP.md](SETUP.md). Code is complete so you can run it first;
for practice, reimplement each kernel in a scratch branch before comparing.
Read [REFERENCES.md](REFERENCES.md) alongside the labs. Each lab can be completed
independently. Run every command in the GPU environment, PowerShell or WSL
([every new terminal](SETUP.md#every-new-terminal)); each writes JSON under
`results/local/`, which `python scripts/report_results.py FILE --output FILE.md`
turns into a readable table.

## 1. Vector addition: learn the execution model

```bash
python examples/hello_triton.py
kernel-bench --op add --output results/local/add.json
```

Read `ops.add`, `_add` in `src/kernel_portfolio/triton_kernels.py`, then
`vector_add` in `cuda/kernels.cu`. In CUDA, a thread computes one scalar index.
In Triton, a program describes a block of indices; the compiler maps those values
to threads. `tl.arange(0, BLOCK)` is a vector of offsets, not a Python loop.

For N=257 and BLOCK=256, grid size is `ceil(257/256)=2`. Program 0 owns elements
0–255. Program 1 owns 256–511 but masks all except 256. Both loads and the store
need masks. A one-program, one-element implementation would waste parallel work.

Experiment: compare block sizes 128/256/512/1024, and tiny versus large vectors.
Predict logical traffic as `3*N*sizeof(dtype)`; intensity is one addition per
three elements transferred. Explain why a different block size cannot remove the
fundamental memory traffic. Exit criterion: draw ownership and explain a losing case.

## 2. Row sum: reductions and numerical order

```bash
kernel-bench --op row_sum --dtype float32 --output results/local/reduction.json
```

One program owns one row. A power-of-two block covers that row; padded elements
must contribute zero. Inputs are converted to FP32 before summation, and the
output remains FP32. The physical row stride is independent of logical width.

Exercise: evaluate width 33, 1024 and 4097. Why does 4097 allocate an 8192-element
logical block? Why can that create register pressure? Why is maximum occupancy
not necessarily maximum throughput? Reproduce a cancellation example such as
`[1e8, 1, -1e8]` and explain why floating-point addition is not associative.
Exit criterion: explain the reduction identity and distinguish algorithmic error
from a harmless change in reduction order.

## 3. Softmax: make stability and fusion visible

```bash
kernel-bench --op softmax --dtype float32 --compile --output results/local/softmax.json
```

Derive `m=max(x)`, `p=exp(x-m)`, `y=p/sum(p)` for each row. Subtracting the
maximum leaves the mathematical probabilities unchanged while avoiding overflow
from large positive logits. Padded lanes load negative infinity; after exponentiation
they contribute zero. A zero padding value would corrupt negative-logit rows.

Compare the eager decomposition, `torch.softmax`, 4-warp and 8-warp Triton
variants, and `torch.compile`. Count intermediates that fusion removes. The
softmax result only needs one input read and one output write in the ideal case;
do not equate that lower bound with measured DRAM traffic.

Run CUDA's serial and parallel implementations. Inspect the warp shuffles,
shared array of eight partial values, and barriers. The final barrier protects
scratch reuse between max and sum reductions. Serial summation uses compensation
after an 8192-wide test exposed normalization drift. Explain how a tree reduction
changes error accumulation. Exit criterion: defend one optimization using the
JSON measurements, explain why a `-inf` (masked) score is safe in the single-block
kernel but needed a guard in the looped one, and explain why rows wider than
8192 switch to a looped kernel whose chunk size depends on the row count.

## 4. Residual RMSNorm: fuse without changing the contract

```bash
kernel-bench --op rmsnorm --compile --output results/local/rmsnorm.json
```

The operation is `z=float32(x)+float32(residual)` followed by
`y=z*rsqrt(mean(z*z)+eps)*float32(weight)`, with one final cast to the input dtype.
It is **not LayerNorm**: it does not subtract a mean. It is also not equivalent
to first rounding the residual addition to FP16. The fused kernel keeps `z`
inside the program instead of writing a full intermediate tensor to device memory.

Exercise: add a second reference with rounded FP16 residuals and measure the
numerical difference. Do not compare implementations with different semantics
as if they were interchangeable. Compare against Inductor before attributing a
large gain to handwritten code. Exit criterion: state the dtype of each intermediate,
show zero-input behavior, and explain the role of epsilon.

## 5. GEMM: tiling, Tensor Cores and tuning costs

```bash
kernel-bench --op matmul --dtype float16 --output results/local/gemm.json
python scripts/inspect_kernel.py --op matmul
```

For `A[M,K] @ B[K,N]`, each program owns a `BM x BN` output tile. It walks K
in chunks of BK, loads `BM x BK` and `BK x BN`, and accumulates their dot product
in FP32. Both input tiles need masks on edge blocks. FP16/BF16 dot products can
map to Tensor Core instructions; inspect PTX rather than assuming a particular
instruction from source syntax alone.

Eight tile/warp/stage/group configurations are compared by the autotuner. Its chosen
configuration is saved with each result. More work per program can improve
reuse while increasing registers/shared memory and reducing scheduling freedom.
Autotuning has a first-call cost and optimizes its own measurement conditions;
the winner can lose to a fixed tile in a separate graph benchmark.

Exercise: draw a ragged 127x255 output with K=65, then sweep a square shape.
Report `2*M*N*K / seconds` and the ratio to `torch.mm`. Keep the case where
cuBLAS wins. Exit criterion: explain why a useful demonstration is not a replacement
for cuBLAS. Then run `python scripts/gemm_grouping.py`, which compares `GROUP_M=1`
(row-major tiles) with grouped order on one tile. On the RTX 4080 SUPER it made no
measurable difference; explain why an L2 larger than B leaves little to recover.

## 6. Profile, inspect, explain

```bash
python scripts/inspect_kernel.py --op softmax
python scripts/profile_kernel.py --op softmax
```

Follow [PROFILING.md](PROFILING.md) and review [CASE_STUDIES.md](CASE_STUDIES.md).
Separate observations from hypotheses:
latency is observed; a claim about bandwidth, occupancy or cache behavior needs
the corresponding evidence. A register count alone does not establish a bottleneck.

Deliver a five-sentence explanation: workload and contract; baseline; proposed
change and reason; correctness and measurements; limitation and next experiment.
Exit criterion: reproduce the results and answer follow-up questions without
memorizing the source. Add your own experiment and commit that work before using
the portfolio to represent your personal experience.
