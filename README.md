# Triton + CUDA Kernel Portfolio

A hands-on path from ordinary Python loops to Triton and CUDA kernels.
Work through small numerical examples, implement
the exercises, then inspect the optimized operators and measurements on an
**NVIDIA RTX 4080 SUPER**.

Read it as a website: **https://buicongnguyen.github.io/triton-cuda-lab/**

**New to kernels? Start with the [beginner course](learning/README.md).**
It has ten lessons, ten editable exercises, hints, separate reference answers,
and a checker that explains which property failed. The first seven exercises
need only Python's standard library.

**Already comfortable?** [Optimization labs](docs/LABS.md) · [GPU setup](docs/SETUP.md) ·
[measured case studies](docs/CASE_STUDIES.md) · [validation](results/VALIDATION.md).

**Next levels:** [Intermediate and advanced workshops](learning/workshops/README.md)
add strided softmax, an explicit backward pass, fused GEMM, online normalization,
wide-row split softmax and streaming attention. Each has a starter, separate
working answer and checks; GPU experiments preserve raw measurements.

## The path, step by step

A session is 45–90 minutes. Do the steps in order; each one links to the guide
that walks you through it.

| Step | What you do | Guide | Terminal | Sessions |
| --- | --- | --- | --- | --- |
| 1 | Indices, strides, reductions, softmax, RMSNorm, GEMM and measurement in plain Python | [Beginner lessons 0–7](learning/README.md) | PowerShell or WSL; no GPU packages | 8–10 |
| 2 | Install the GPU environment once | [SETUP.md](docs/SETUP.md) | PowerShell (or WSL) | 1 |
| 3 | Write three Triton kernels, then read the CUDA version | [Lessons 8–9](learning/lessons/08_triton.md) | GPU environment; PowerShell to build CUDA | 3–5 |
| 4 | Change configurations and explain measurements | [Labs](docs/LABS.md), [case studies](docs/CASE_STUDIES.md) | GPU environment | 3–6 |
| 5 | Harder kernels: views, backward, fused GEMM, online softmax, attention | [Workshops](learning/workshops/README.md) | GPU environment (A1 is CPU-only) | 2–5 each |
| 6 | Explain your own work | [Journal](learning/JOURNAL.md), [case studies](docs/CASE_STUDIES.md) | Any | 2 |

The [18-session schedule](learning/SCHEDULE.md) breaks steps 1–4 into sessions.
Stuck on a word such as *stride*, *L2*, *CUDA graph* or *constexpr*? See the
[glossary](learning/GLOSSARY.md).

### Which terminal? (Windows)

| Task | Use | Why |
| --- | --- | --- |
| Lessons 0–7, workshop A1, `--hint`, `learning.explain` | Any PowerShell with `python` (or WSL with `python3`) | Standard library only |
| Anything using Triton: lesson 8, GPU workshops, `kernel-bench`, `tests/` | The **GPU environment**: PowerShell after `.\.venv-win\Scripts\Activate.ps1`, or WSL after `source .venv/bin/activate` ([setup](docs/SETUP.md)) | Needs PyTorch with CUDA plus Triton (`triton-windows` on Windows) |
| Building and running `cuda/kernels.cu` | PowerShell with `scripts\build_cuda_windows.cmd`, or CMake on Linux | Uses the Windows CUDA Toolkit and Visual Studio |

Everything can be done from PowerShell. WSL is an alternative for the GPU work, and
it is what produced the recorded results. A checker line reading `UNAVAILABLE`
(exit code 3) means the current terminal has no PyTorch, Triton or NVIDIA GPU. It is
not a failure in your code: activate the GPU environment
([every new terminal](docs/SETUP.md#every-new-terminal)).

## First session — no GPU needed

From the repository root:

```bash
python -m learning.check --list
python -m learning.explain indexing
# Edit learning/exercises/offsets.py, then check your answer:
python -m learning.check offsets
```

Read [offsets and masks](learning/lessons/01_indexing.md) alongside the exercise.
An unfinished starter reports `TODO`; that is expected. Use `--hint` when stuck
and `--solution` only to check the separate reference answer.

| Learn first | Then implement | Finally explain |
| --- | --- | --- |
| Indices and masks | Ceiling division and block ownership | Why N=257 needs a tail mask |
| Storage and strides | Read cropped and transposed views | Why shape alone cannot locate a row |
| Reduction trees | Pairwise summation with neutral padding | Why FP32 accumulation and order matter |
| Stable softmax | Max → shift → exp → sum → divide | Why zero padding gives wrong probabilities |
| RMSNorm and GEMM | Scalar formulas and tile ranges | Precision semantics and input reuse |
| Measurement | Speedup, GB/s and Amdahl's law | Why graph and Python-dispatch results differ |
| Triton and CUDA | Three GPU exercise kernels | How block-level values map to thread cooperation |

Use the [18-session schedule](learning/SCHEDULE.md), [answer explanations](learning/ANSWER_KEY.md)
and [personal journal](learning/JOURNAL.md) to turn each lesson into evidence you can discuss.

## What is implemented

| Stage | Code | Optimization question |
| --- | --- | --- |
| 1. Vector add | [`add`](src/kernel_portfolio/ops.py), [`hello_triton.py`](examples/hello_triton.py) | How do masks, coalescing and block size affect a bandwidth-heavy kernel? |
| 2. Row sum | [`_row_sum`](src/kernel_portfolio/triton_kernels.py) | How do reduction order, row width and accumulation dtype matter? |
| 3. Softmax | [`_softmax`](src/kernel_portfolio/triton_kernels.py) | When does fusion beat eager and compiled PyTorch? |
| 4. Residual RMSNorm | [`_rmsnorm`](src/kernel_portfolio/triton_kernels.py) | Can an intermediate tensor be eliminated without changing precision semantics? |
| Wide rows | [`_row_plan`, looped and split kernels](src/kernel_portfolio/triton_kernels.py) | How do rows wider than one block (up to 2^20) stay fast, and when must a row be split across programs? |
| Training | [`_softmax_bwd`, `_rmsnorm_bwd`](src/kernel_portfolio/triton_kernels.py), [autograd registration](src/kernel_portfolio/library.py) | What does a backward kernel need, and how does a custom op train inside `torch.compile`? |
| 5. GEMM | [`_matmul`](src/kernel_portfolio/triton_kernels.py) | When do grouped tile order and autotuning beat a fixed configuration or cuBLAS? |
| torch.compile | [`library.py`](src/kernel_portfolio/library.py) | How do custom ops let a compiled model replay many kernels as one CUDA graph? |
| CUDA comparison | [`kernels.cu`](cuda/kernels.cu) | How do threads, warp shuffles, shared-memory barriers, online statistics and `float4` loads implement the same ideas? |

Each operator has independent reference checks. Benchmarks keep raw samples,
correctness errors, hardware/software metadata and source hashes. Results include
losing cases; no universal speedup is claimed. See the [measurement method](docs/BENCHMARKING.md).

## GPU portfolio quick start

Native Windows PowerShell and Ubuntu WSL2 both work; [SETUP.md](docs/SETUP.md)
compares them and has native CUDA build commands. In PowerShell:

```powershell
.\scripts\setup_windows.ps1              # once: creates .venv-win (about 3 GB)
.\.venv-win\Scripts\Activate.ps1        # in each new terminal
python examples/hello_triton.py
python -m learning.check all --include-gpu --solution
$env:KERNEL_REQUIRE_GPU = "1"; python -m unittest discover -s tests -v
kernel-bench --op softmax --dtype float32 --compile --output results/local/softmax.json
```

In WSL, after the [WSL setup](docs/SETUP.md#one-time-setup-wsl), the same commands
apply, with `source .venv/bin/activate` and `KERNEL_REQUIRE_GPU=1 python -m unittest ...`.

`softmax` and `residual_rmsnorm` **train**: given a gradient-tracking input, they run
through custom ops whose backward passes are Triton kernels, in eager code and inside
`torch.compile`. `add`, `row_sum` and `matmul` are forward-only and reject
gradient-tracking inputs while autograd is enabled (parameters work under
`torch.no_grad()`). All reject mixed devices, unsupported layouts and out-of-range
shapes. Row kernels support FP32/FP16/BF16 and widths 1–1,048,576 (rows wider than
8192 loop over chunks, or split across programs when there are fewer rows than SMs);
GEMM supports contiguous FP16/BF16 inputs with FP32 accumulation. For use inside
`torch.compile`, `import kernel_portfolio.library` registers the same kernels as
`torch.ops.kernel_portfolio.*` custom ops. Read the full
[contracts](docs/SETUP.md#public-operator-contracts) before using them.

## Document map

**Guides to learn from, in reading order**

- [Beginner course](learning/README.md): lessons, exercises, [answer explanations](learning/ANSWER_KEY.md),
  [schedule](learning/SCHEDULE.md), [journal template](learning/JOURNAL.md) and [glossary](learning/GLOSSARY.md).
- [Workshops](learning/workshops/README.md): six deeper tasks, their
  [experiment worksheet](learning/workshops/EXPERIMENTS.md) and [answer explanations](learning/workshops/ANSWER_KEY.md).

**Guides for running and measuring**

- [SETUP.md](docs/SETUP.md): environments, GPU and CUDA builds, operator contracts, troubleshooting.
- [LABS.md](docs/LABS.md): six experiments on the finished operators.
- [BENCHMARKING.md](docs/BENCHMARKING.md): what the timing modes measure and what they exclude.
- [PROFILING.md](docs/PROFILING.md): compiler IR/PTX, Nsight and Compute Sanitizer.
- [CASE_STUDIES.md](docs/CASE_STUDIES.md) and [results/SUMMARY.md](results/SUMMARY.md): measured results and their interpretation.

**Reading**

- [REFERENCES.md](docs/REFERENCES.md) and [book previews](references/books/README.md):
  Manning's *GPU Programming with Triton*, Fregly's *AI Systems Performance
  Engineering* and official Triton/NVIDIA documentation.

**Project records (optional: how the repo was planned, reviewed and validated)**

- [BEGINNER_UPGRADE.md](docs/records/BEGINNER_UPGRADE.md) and
  [INTERMEDIATE_ADVANCED_PLAN.md](docs/records/INTERMEDIATE_ADVANCED_PLAN.md): design plans.
- [REVIEW.md](docs/records/REVIEW.md): review findings and fixes, including why the results were re-measured.
- [results/VALIDATION.md](results/VALIDATION.md): exactly what was executed, with logs.

## Evidence and limits

The [results folder](results/) contains actual local GPU runs, not estimated
numbers. GPU tests cover ragged dimensions, dtype differences, row strides,
empty inputs, stable softmax, invalid calls, a non-default stream and kernel reuse
across shapes. CPU CI checks reference math and contracts on Python 3.10 and 3.12,
and runs the Triton kernels in Triton's CPU interpreter; it does not certify GPU
behavior or performance.

The teaching path's reference solutions passed **40 CPU checks and 36 GPU checks**;
the checker has its own regression tests. See [beginner validation](results/learning/VALIDATION.md).
Reference checks validate the material and do not mark your unfinished exercises as complete.

This is an AI-assisted learning project. Reproduce the experiments and add your
own explanations before presenting any of it as your own work.
Autograd for `add`, `row_sum` and `matmul`, double backward, batched attention,
non-NVIDIA accelerators, distributed collectives and production model integration
are future work. Softmax and residual RMSNorm train through Triton backward kernels;
the workshops add an explicit softmax VJP exercise and bounded single-head
streaming attention. Hardware-counter profiling requires permissions
unavailable in the local validation environment; that limitation is recorded.

## License

MIT; see [LICENSE](LICENSE).
