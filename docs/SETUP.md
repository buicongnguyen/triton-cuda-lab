# Setup and execution

The CPU lessons (0–7) and workshop A1 run in any Python 3.10+ terminal. Everything
that imports Triton needs a GPU environment. On Windows you can choose either
option below; both were validated on this machine with the same tests. Command words
you do not recognize are in the [glossary](../learning/GLOSSARY.md).

| | Option A: PowerShell (native Windows) | Option B: Ubuntu in WSL2 |
| --- | --- | --- |
| Terminal | The PowerShell you already use | Ubuntu, opened from PowerShell with `wsl` |
| Triton package | `triton-windows` 3.6.0, a community build | `triton` 3.6.0 from the Triton project |
| Setup effort | One script | Five short steps |
| Validated | All tests and reference checks pass | All tests and reference checks pass; produced the recorded results |
| Choose it when | You want one terminal for everything | You want the upstream, Linux-only toolchain |

Upstream Triton publishes Linux packages only; `triton-windows` is maintained
outside the Triton project and follows its releases. On this machine it measured
close to WSL, e.g. FP16 softmax 16384x4096: 441 us in PowerShell vs 425 us in WSL.
Compare timings only within one environment.

## Get the code

Clone the repository once. The folder it creates is the *repo root* that later
steps refer to; if you clone it under another name or location, use that path.

```powershell
cd C:\Users\<you>\source\repos
git clone https://github.com/buicongnguyen/triton-cuda-lab.git
cd triton-cuda-lab
```

## Option A: Windows PowerShell

### One-time setup (PowerShell)

1. **Check the prerequisites.** `nvidia-smi` should list the RTX 4080 SUPER, and
   `python --version` should print 3.10 or newer.

2. **Run the setup script** from the repo root. It creates `.venv-win` (ignored by
   Git), installs PyTorch with CUDA, `triton-windows`, NumPy and this repo, then
   prints an environment report. The download is about 3 GB. Re-running it is
   safe; deleting `.venv-win` removes everything it installed.

   ```powershell
   .\scripts\setup_windows.ps1
   ```

3. **Verify it.** `kernel-doctor` should report `"cuda_available": true` and a
   Triton version. The test suite should end with `OK`; one two-GPU test and three
   interpreter-only tests are expected to be skipped.

   ```powershell
   .\.venv-win\Scripts\Activate.ps1
   kernel-doctor
   $env:KERNEL_REQUIRE_GPU = "1"; python -m unittest discover -s tests -v
   python -m learning.check triton_add --solution
   ```

### PowerShell versions of bash commands

Most commands in these docs, such as `python -m ...` and `kernel-bench ...`, are
identical in both shells. The exceptions:

| Bash (Linux/WSL) | PowerShell |
| --- | --- |
| `KERNEL_REQUIRE_GPU=1 python -m unittest ...` | `$env:KERNEL_REQUIRE_GPU = "1"; python -m unittest ...` |
| `TRITON_INTERPRET=1 python ...` | `$env:TRITON_INTERPRET = "1"; python ...`, then `Remove-Item Env:TRITON_INTERPRET` |
| `CUDA_LAUNCH_BLOCKING=1 python ...` | `$env:CUDA_LAUNCH_BLOCKING = "1"; python ...` |
| `export PYTHONPATH=src` | `$env:PYTHONPATH = "src"` |
| `source .venv/bin/activate` | `.\.venv-win\Scripts\Activate.ps1` |
| `mkdir -p results/local` | `New-Item -ItemType Directory -Force results\local` |
| A line ending in `\` continues on the next line | End the line with a backtick `` ` `` instead, or join the lines |

In PowerShell, `$env:NAME = "..."` stays set until the terminal closes. That is
harmless for `KERNEL_REQUIRE_GPU`, but remove `TRITON_INTERPRET` after use: while it
is set, every Triton kernel runs slowly on the CPU instead of the GPU.

## Option B: Linux or Ubuntu in WSL2

The NVIDIA Windows driver exposes the GPU to WSL. The CUDA version printed by
`nvidia-smi` is a driver capability, not the version of an installed Python wheel
or compiler.

### One-time setup (WSL)

1. **Open Ubuntu** from PowerShell. If the distribution is missing, install it
   first with `wsl --install -d Ubuntu-22.04` and restart when asked.

   ```powershell
   wsl -d Ubuntu-22.04
   ```

2. **Go to the checkout.** Windows drive `C:\` appears as `/mnt/c/` inside WSL:

   ```bash
   cd /mnt/c/Users/<your-windows-user>/source/repos/triton-cuda-lab
   ```

   For faster builds, a separate checkout on the Linux filesystem can be useful.

3. **Confirm WSL sees the GPU.** `nvidia-smi` should list the RTX 4080 SUPER. The
   Windows NVIDIA driver provides this; do not install a Linux GPU driver inside WSL.

   ```bash
   nvidia-smi
   ```

4. **Create the Python environment** (Python 3.10 or newer, CUDA-enabled PyTorch):

   ```bash
   python3 -m venv .venv                  # Ubuntu may first need: sudo apt install python3-venv
   source .venv/bin/activate              # the prompt now starts with (.venv)
   python -m pip install --upgrade pip
   # requirements-tested.txt pins the versions validated together.
   python -m pip install -c requirements-tested.txt torch --index-url https://download.pytorch.org/whl/cu130
   python -m pip install -c requirements-tested.txt -e '.[gpu,dev]'  # repo, Triton, Ruff, NumPy
   ```

5. **Verify it.** Same expectations as in option A:

   ```bash
   kernel-doctor
   KERNEL_REQUIRE_GPU=1 python -m unittest discover -s tests -v
   python -m learning.check triton_add --solution
   ```

Select a PyTorch wheel compatible with your driver if your system differs.
`requirements-tested.txt` lists the versions validated together for Linux and
Windows; CI and the Windows setup script install with it as a constraints file.
It is not a complete package lock or a claim that every allowed version was tested.
Triton's runtime toolchain comes with its wheel; building standalone CUDA also
requires the CUDA Toolkit and a supported C++ host compiler. The recorded results
used an existing WSL virtual environment with the tested versions and
`PYTHONPATH=src`; any environment matching `requirements-tested.txt` is equivalent.

## Every new terminal

The GPU environment is not remembered between terminals. Each time, from the repo root:

| PowerShell (option A) | WSL (option B) |
| --- | --- |
| `cd C:\Users\<you>\source\repos\triton-cuda-lab` | `wsl -d Ubuntu-22.04`, then `cd /mnt/c/Users/<you>/source/repos/triton-cuda-lab` |
| `.\.venv-win\Scripts\Activate.ps1` | `source .venv/bin/activate` |

The prompt then starts with `(.venv-win)` or `(.venv)`. If you use an environment
without `pip install -e .`, set `PYTHONPATH` to `src` (see the table above) and
replace the two installed commands with their module forms:

| Installed command | Equivalent without installing |
| --- | --- |
| `kernel-bench ...` | `python -m kernel_portfolio.benchmark ...` |
| `kernel-doctor` | `python -m kernel_portfolio.environment` |

Benchmarks write JSON wherever `--output` points; `results/local/` is ignored by
Git, so it is the place for your own runs. To read a run as a table:

```bash
python scripts/report_results.py results/local/my-run.json --output results/local/my-run.md
```

## CPU-only checks

Install a CPU PyTorch wheel and this package; Triton is optional for imports,
reference math and contract checks:

```bash
python -m pip install -c requirements-tested.txt torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c requirements-tested.txt -e '.[dev]'
python -m unittest discover -s tests -v   # GPU and interpreter tests skip themselves
```

Running all tests without a GPU explicitly skips the GPU class. For a GPU release
gate, **always set `KERNEL_REQUIRE_GPU=1`** so a missing device fails the run.

Triton's interpreter executes the kernel code on CPU tensors, without a GPU. It
checks indexing, masks and math, not GPU timing or races. CI runs it on Linux; it
also works in both GPU environments above. Triton 3.6's interpreter needs NumPy
older than 2.4; the `dev` extra and the constraints file provide it:

```bash
python -m pip install -c requirements-tested.txt -e '.[gpu,dev]'   # option A already has both
TRITON_INTERPRET=1 python -m unittest discover -s tests -p 'test_interpreter.py' -v
```

In PowerShell: `$env:TRITON_INTERPRET = "1"; python -m unittest discover -s tests -p test_interpreter.py -v; Remove-Item Env:TRITON_INTERPRET`.

## Standalone CUDA C++

Native Windows, from PowerShell at the repo root. The script needs Visual Studio's
"Desktop development with C++" workload and the CUDA Toolkit (which sets
`CUDA_PATH`); it prints what is missing if either is not found.

```powershell
.\scripts\build_cuda_windows.cmd
.\build\cuda_portfolio.exe --test-only
New-Item -ItemType Directory -Force results\local | Out-Null
.\build\cuda_portfolio.exe --json results\local\cuda.json
```

`--test-only` should print `CUDA correctness passed (3 vector + 18 softmax cases)`.
The script builds into the repo's `build` folder from any directory. It targets
`sm_89`, the RTX 4080 SUPER; on another GPU set the architecture first, for example
`$env:CUDA_ARCH = "sm_86"` for an RTX 30-series card.

Linux with an installed CUDA Toolkit:

```bash
cmake -S cuda -B build -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
mkdir -p results/local
./build/cuda_portfolio --json results/local/cuda.json
```

Set the architecture to your GPU's compute capability; 89 is the RTX 4080 SUPER
tested here. This CMake path is supplied for portability; the local CUDA
executable was built natively on Windows.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `python` opens the Microsoft Store or prints "Python was not found" | Windows app-execution alias, no real Python | Install Python from python.org, or turn off the alias under Settings > Apps > Advanced app settings > App execution aliases |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | PowerShell execution policy | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then open a new terminal |
| `No module named learning` | Terminal is not at the repo root | `cd` to the folder that contains `README.md` |
| `UNAVAILABLE: torch is not installed` or `triton is not installed` | The GPU environment is not active in this terminal | Activate it as in [every new terminal](#every-new-terminal) |
| `UNAVAILABLE: an NVIDIA CUDA GPU is required` | CPU-only PyTorch wheel, or WSL cannot see the GPU | Run `nvidia-smi`; reinstall PyTorch with the CUDA index URL above |
| `No module named kernel_portfolio` or `kernel-bench` is not recognized | This repo is not installed in the active environment | `python -m pip install -e .`, or set `PYTHONPATH` to `src` and use the module forms above |
| `ensurepip is not available` when creating `.venv` in WSL | Ubuntu's venv package is missing | `sudo apt install python3-venv` |
| The first GPU call pauses for a second or more | Triton compiles the kernel (and autotunes GEMM) on first use, then caches it under `.triton/cache` in your home folder | Normal. Benchmarks warm up before timing |
| Every Triton call is suddenly very slow | `TRITON_INTERPRET` is still set, so kernels run on the CPU | PowerShell: `Remove-Item Env:TRITON_INTERPRET`; bash: `unset TRITON_INTERPRET` |
| Interpreter tests fail with `only 0-dimensional arrays can be converted to Python scalars` | NumPy 2.4 or newer | `python -m pip install "numpy<2.4"` |
| The checker prints `TODO` | The starter still contains `raise NotImplementedError` | Expected until you implement the function |
| `CUDA error: an illegal memory access` | A kernel read or wrote out of bounds; the process is now unusable | Start a new process; rerun with `CUDA_LAUNCH_BLOCKING=1` and `--verbose`; check masks and strides |
| Build script prints `vswhere was not found` or `CUDA_PATH is not set` | Visual Studio C++ tools or CUDA Toolkit missing | Install the missing one, then open a new terminal |

Checker exit codes: 0 all passed, 1 a wrong answer or load error, 2 unfinished
(`TODO`), 3 nothing could run (`UNAVAILABLE`).

## Use the operators from Python

In the GPU environment:

```python
import torch
from kernel_portfolio import add, matmul, residual_rmsnorm, row_sum, softmax

x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
y = softmax(x)                                   # same shape and dtype as x
w = torch.ones(1024, device="cuda", dtype=torch.float16)
z = residual_rmsnorm(x, x, w)                    # eps defaults to 1e-5
c = matmul(x, x)                                 # FP16/BF16 only
with torch.no_grad():                            # model parameters are fine here
    z = residual_rmsnorm(x, x, torch.nn.Parameter(w))

# softmax and residual_rmsnorm train: autograd runs their Triton backward kernels.
weight = torch.nn.Parameter(torch.ones(1024, device="cuda", dtype=torch.float16))
h = x.detach().requires_grad_()
softmax(residual_rmsnorm(h, h, weight)).float().square().sum().backward()
print(h.grad.shape, weight.grad.shape)           # both gradients are filled
```

The backward kernels are also callable directly, for example to measure them:
`softmax_backward(grad, y)` takes the saved softmax output, and
`residual_rmsnorm_backward(grad, x, residual, weight)` returns the input gradient
(shared by `x` and `residual`) and the weight gradient.

Unsupported inputs raise `ValueError` or `TypeError` before anything launches;
the message names the violated rule.

Inside a compiled model, use the custom-op versions. They run the same kernels
with the same shape, dtype and device checks, and `torch.compile` can capture them
in one graph. The softmax and residual RMSNorm custom ops register their backward
kernels with autograd, so a compiled training step traces both directions. The
`add`, `row_sum` and `matmul` custom ops have no autograd formula: they accept
gradient-tracking inputs but raise on backward. For inference, use
`torch.no_grad()` or `torch.inference_mode()`:

```python
import kernel_portfolio.library  # registers torch.ops.kernel_portfolio.*

def block(x, r, w):
    return torch.ops.kernel_portfolio.softmax(torch.ops.kernel_portfolio.residual_rmsnorm(x, r, w))

fast_block = torch.compile(block, mode="reduce-overhead")  # replays as a CUDA graph
with torch.no_grad():
    y = fast_block(x, x, w)
```

Use the plain `kernel_portfolio.*` functions for eager calls: going through
`torch.ops` adds dispatcher cost, and compiling a single tiny op does not pay off
(see the [case study](CASE_STUDIES.md#torchcompile-pay-dispatch-once-per-graph)).

## Public operator contracts

| Operator | Inputs | Output / limitations |
| --- | --- | --- |
| `add` | Equal contiguous vectors, FP32/FP16/BF16 | Same shape/dtype; empty vector allowed |
| `row_sum` | 2D, width 1–1,048,576, contiguous columns (any stride for width 1), nonoverlapping rows | One FP32 value per row; rows wider than 8192 loop over chunks |
| `softmax` | Same row layout; finite values, or `-inf` for masked entries | Same shape/dtype, stable along last dimension; masked entries get 0, and a row with no finite value gives NaN as in PyTorch. Records gradients |
| `softmax_backward` | Upstream gradient of any strides; the contiguous softmax output | Input gradient `y * (grad - sum(y * grad))`, same dtype |
| `residual_rmsnorm` | Same-shape row tensors and contiguous weight vector of matching dtype/device | FP32 residual addition/reduction, one rounding at output, positive finite epsilon. Records gradients for all three tensors |
| `residual_rmsnorm_backward` | Upstream gradient of any strides; the forward's inputs and `eps` | Input gradient (for both `x` and `residual`) and weight gradient; the weight gradient is summed over rows in FP32 in a fixed order, so it is deterministic |
| `matmul` | Contiguous 2D FP16/BF16, matching inner dimensions | Same dtype, FP32 accumulation; zero dimensions supported |

All calls require NVIDIA CUDA, reject mixed dtypes/devices, and allocate fresh
outputs. `add`, `row_sum` and `matmul` reject `requires_grad` inputs while autograd
is enabled (their custom ops accept them but raise on backward); `softmax` and
`residual_rmsnorm` record gradients instead, and double backward is not supported.
Both interfaces accept model parameters under `torch.no_grad()` or
`torch.inference_mode()`. Rows wider than 8192 loop over chunks in one program per
row, except that softmax and RMSNorm (above 16384) split each row across programs
when there are fewer rows than SMs.
Row width and GEMM N/K are compile-time constants: the first call with a new width
or weight shape compiles, typically once per model. Vector lengths, row counts,
row strides, GEMM M and `eps` are runtime arguments, so a new batch size reuses the
compiled kernel. GEMM autotuning runs once per power-of-two bucket of M and times
real launches, so call `matmul` once for a new bucket before capturing it in a
CUDA graph.

No implicit input copies, broadcasting, double backward, NaN/Inf policy, or production
dispatcher is provided. Inputs must be finite, except that softmax accepts `-inf`
for masked entries; RMSNorm also assumes its FP32 squared residuals and reduction
remain finite. The wrappers do not scan tensors
for finite values, since that would add launches. The row-width cap bounds
resource usage and is not a hardware maximum. Physical tensor address spans and
GEMM output sizes must fit signed 32-bit indexing. These are explicit
teaching-kernel bounds, checked before launch.
