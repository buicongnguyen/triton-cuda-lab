# Start here: learn to explain and write a GPU kernel

You only need basic Python to begin: lists, loops, functions and arithmetic.
The first seven exercises use **no PyTorch, no Triton and no GPU**. You will first
work out the same indexing and mathematics with small lists, then translate
three operations into real GPU kernels.

Use a terminal opened at the repository root (the folder containing the top-level
`README.md`), not the `learning/` folder. Every command is run from there.

**Which terminal?** Lessons 0–7 need only Python 3.10+, so any PowerShell with
`python` works (Ubuntu WSL with `python3` does too). Lesson 8 and every GPU step
need the GPU environment from the [setup guide](../docs/SETUP.md): on Windows,
run `.\scripts\setup_windows.ps1` once, then `.\.venv-win\Scripts\Activate.ps1`
in each new PowerShell terminal. WSL is an equivalent alternative.
If `python` opens the Microsoft Store, see [troubleshooting](../docs/SETUP.md#troubleshooting).

New words are defined in the [glossary](GLOSSARY.md), grouped by the lesson where
you first need them.

## Your first 20 minutes

1. Run `python -m learning.check --list` to see the sequence.
2. Read [lesson 0](lessons/00_orientation.md), then [lesson 1](lessons/01_indexing.md).
3. Run `python -m learning.explain indexing` and predict its last four lines.
4. Open [exercises/offsets.py](exercises/offsets.py) and replace the two
   `NotImplementedError` statements with your code.
5. Run `python -m learning.check offsets` until every case passes.
6. Write one sentence explaining why offsets 10 and 11 must not load memory.

That is a complete first session. You do not need to understand Tensor Cores yet.

## Course map

| Order | Lesson | Edit this file | Check command | Ready when you can… |
| --- | --- | --- | --- | --- |
| 0 | [Setup and the execution model](lessons/00_orientation.md) | None | `python -m learning.check --list` | Explain host, device, tensor and kernel |
| 1 | [Offsets and masks](lessons/01_indexing.md) | [offsets.py](exercises/offsets.py) | `python -m learning.check offsets` | Draw the final partial block |
| 2 | [Storage, shapes and strides](lessons/02_strides.md) | [strides.py](exercises/strides.py) | `python -m learning.check strides` | Find a sliced tensor's actual storage index |
| 3 | [Reduction trees](lessons/03_reductions.md) | [reduction.py](exercises/reduction.py) | `python -m learning.check reduction` | Choose neutral padding and show each reduction level |
| 4 | [Stable softmax](lessons/04_softmax.md) | [softmax.py](exercises/softmax.py) | `python -m learning.check softmax` | Calculate all intermediates in a three-element row |
| 5 | [Residual RMSNorm](lessons/05_rmsnorm.md) | [rmsnorm.py](exercises/rmsnorm.py) | `python -m learning.check rmsnorm` | Distinguish RMSNorm from LayerNorm and explain precision |
| 6 | [Matrix multiplication and tiles](lessons/06_matmul.md) | [matmul.py](exercises/matmul.py) | `python -m learning.check matmul` | Draw output ownership and the K loop |
| 7 | [Measure and interpret](lessons/07_measurement.md) | [measurement.py](exercises/measurement.py) | `python -m learning.check measurement` | Explain a speedup with its baseline, units and timing mode |
| 8 | [Write your first three Triton kernels](lessons/08_triton.md) | [triton_add.py](exercises/triton_add.py), [triton_sum.py](exercises/triton_sum.py), [triton_softmax.py](exercises/triton_softmax.py) | `python -m learning.check triton_add` (then `triton_sum`, `triton_softmax`) | Translate your Python reasoning into GPU loads, reductions and stores |
| 9 | [Understand the CUDA counterpart](lessons/09_cuda.md) | Read `cuda/kernels.cu` | Run the built CUDA executable with `--test-only` | Explain threads, warps, shared partials and barriers |

Lessons 5, 6 and 7 each end with an optional benchmark marked **GPU step**. Skip
it on your first pass and return after lesson 8's setup. The
[schedule](SCHEDULE.md) does exactly that.

CPU checks: `python -m learning.check all`. CPU + GPU:
`python -m learning.check all --include-gpu`. The GPU path requires the environment
from [SETUP.md](../docs/SETUP.md).

## When you get stuck

```bash
python -m learning.check softmax --hint
python -m learning.explain softmax
python -m learning.check softmax --verbose
# After your own attempt, compare against the separate reference answer:
python -m learning.check softmax --solution
```

What each checker word means:

| Output | Meaning | What to do |
| --- | --- | --- |
| `PASS name` | This property holds | Nothing |
| `TODO name: message` | The function still raises `NotImplementedError` | Implement it; the message is a hint |
| `FAIL name: AssertionError: expected ..., got ...` | Your code ran but returned a wrong value | Compare with the lesson's worked example |
| `LOAD ERROR` | Your file has a syntax error or crashes on import | Read the error line; `--verbose` shows the traceback |
| `UNAVAILABLE` | This terminal has no PyTorch/Triton/GPU | Activate the GPU environment ([every new terminal](../docs/SETUP.md#every-new-terminal)) |

The checker prints the file it loaded. `--solution` tests reference code and
prints **REFERENCE SOLUTIONS (not learner completion)**. It never edits your files.
Exercise files and your journal are excluded from the maintained course manifest;
completing them does not require refreshing hashes or rerunning reference benchmarks.
Exit codes are 0 for checks passed, 1 for a wrong answer or load error, 2 for an
unfinished exercise, and 3 for unavailable GPU dependencies. A command-line usage
error also returns 2. An unfinished starter is expected; it is not a broken repo.

After CPU lessons, use the [answer explanations](ANSWER_KEY.md),
[18-session schedule](SCHEDULE.md) and [learning journal](JOURNAL.md). The checker
tests behavior on selected cases, not whether you understand or optimized the code.
Explain your result aloud before moving on.

## Beyond the exercises

Continue with the [intermediate and advanced workshops](workshops/README.md).
They add six deeper tasks with separate solutions: real tensor layouts, backward,
fused GEMM, mergeable softmax statistics, wide rows and tiled attention.

The exercise kernels have narrow, stated input assumptions and minimal wrappers
so the math stays visible. The [portfolio implementations](../src/kernel_portfolio/ops.py)
add validation, dtype rules and device handling. Compare those wrappers only after
your first kernel works. Then run the [optimization labs](../docs/LABS.md) and
read the [actual case studies](../docs/CASE_STUDIES.md). Those are measured examples
to investigate, not speedups you are expected to reproduce exactly.
