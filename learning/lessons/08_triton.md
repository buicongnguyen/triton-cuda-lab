# 8. Translate your reasoning into three GPU kernels

**Time:** three sessions of 60–90 minutes. **Prerequisite:** lessons 0–6 and a
working [GPU environment](../../docs/SETUP.md): PowerShell with `.venv-win`
activated, or WSL with `.venv` activated.

**Before you edit anything,** confirm the environment by checking the reference
answer. It should pass. If it does not, the problem is the setup, not your code:
see [troubleshooting](../../docs/SETUP.md#troubleshooting).

```bash
python -m learning.check triton_add --solution
```

You have already solved the hard conceptual pieces with Python lists. Now learn
the small set of Triton operations that express those ideas on a GPU.

## Vocabulary at the code boundary

| Python exercise idea | Triton spelling | Meaning |
| --- | --- | --- |
| A function that runs on the GPU | `@triton.jit` | Specialize and compile a kernel |
| Program number | `tl.program_id(0)` | ID along grid dimension 0 |
| Local positions 0..BLOCK-1 | `tl.arange(0, BLOCK)` | A block of integer offsets |
| Read selected storage values | `tl.load(pointer, mask, other)` | Masked device memory load |
| Write selected values | `tl.store(pointer, value, mask)` | Masked device memory store |
| Change calculation precision | `.to(tl.float32)` | Convert values before arithmetic |
| One sum of a block | `tl.sum(values, axis=0)` | Parallel reduction |
| One maximum of a block | `tl.max(values, axis=0)` | Parallel reduction |
| Exponentiate each value | `tl.exp(values)` | Elementwise approximate exponential |
| Known specialization parameter | `BLOCK: tl.constexpr` | Compiler sees this value; each new value compiles a new kernel |
| Size or stride that varies per call | plain `N`, `STRIDE` arguments | Runtime integers; a new shape reuses the compiled kernel |

For a one-dimensional row block, `axis=0` means its only axis. It does not mean
“reduce rows of the original matrix”; each program already loaded just one row.

`BLOCK` must be `tl.constexpr` because `tl.arange` needs it at compile time. For
every other size, ask whether it changes from call to call. A vector length, a row
count or a batch size does, so it belongs in a runtime argument; as `constexpr`,
every new value would trigger a fresh compile. The exercises pass every size at
runtime, which is always correct. The portfolio kernels in `triton_kernels.py` go
one step further: a model's row width is fixed, so they mark the width `constexpr`.
That costs one compile per width and measured up to 1.9x faster for widths that do
not fill their power-of-two block (see the
[case study](../../docs/CASE_STUDIES.md#specialize-what-the-model-fixes)). Triton
also specializes runtime integers that equal 1 or are divisible by 16, which
preserves alignment hints.

## A. Vector addition

Open [triton_add.py](../exercises/triton_add.py). The wrapper is already provided:

```python
out = torch.empty_like(x)
add_kernel[(triton.cdiv(x.numel(), 256),)](x, y, out, x.numel(), 256)
```

`out` allocates storage but does not initialize the values. The kernel must write
every valid output. The brackets select the grid; the parentheses pass arguments.
For 257 elements, `(2,)` is the one-dimensional grid: two program instances.

Complete the four commented steps inside `add_kernel`. Then remove the
`NotImplementedError` line from `add`; keep its allocation and launch code.

```bash
python -m learning.check triton_add
```

The first run pauses briefly while Triton compiles your kernel; later runs reuse
the compiled version. If the checker prints `UNAVAILABLE`, the GPU environment is
not active; see [every new terminal](../../docs/SETUP.md#every-new-terminal).

The checker tests FP32 and FP16 at 0, 1, 257, 4097 and 65537 elements. It also
checks that your kernel did not overwrite the inputs. An empty input should
return an empty output without launching a zero-sized grid.

## B. Row sum

Open [triton_sum.py](../exercises/triton_sum.py). Now the grid is `(rows,)`:
program 0 owns row 0, program 1 owns row 1, and so on. Program count no longer
depends on a vector's total element count.

For a width of 33, `triton.next_power_of_2(33)` supplies BLOCK=64. Thirty-one
padding lanes load zero. The output has shape `[rows]`, not `[rows,width]`.
The input might have gaps between rows, so use `STRIDE`, not N, for its row offset.

```bash
python -m learning.check triton_sum
```

Finish the kernel, then remove the wrapper guard. The checker expects FP32 output,
including when the input is FP16. Compare to `_row_sum` in the main implementation
only after your attempt.

## C. Stable softmax

Open [triton_softmax.py](../exercises/triton_softmax.py). Start from the load
pattern you just used. Change the padding identity to negative infinity; use
two reductions and an elementwise exponential; store a full row of probabilities.

The input address is `X + row * STRIDE + column`. The freshly allocated output
is contiguous, so its address is `OUT + row * N + column`. Input and output
strides are not automatically the same.

```bash
python -m learning.check triton_softmax
python -m learning.check all --include-gpu
```

Tests include row gaps, non-power-of-two widths and large positive/negative constant
rows. If only the row-gap test fails, inspect addressing before adjusting tolerances.
If negative constant rows fail, inspect the padding identity and stabilization.

Convert the loaded values to FP32 here too, even though the checker cannot tell.
In the row sum, a sum of FP16 values comes back as FP16 and is rounded before the
FP32 store, so a missing conversion fails. In softmax, Triton 3.6 already evaluates
`tl.max`, `tl.exp` and `tl.sum` of FP16 values in FP32 (the generated IR converts
with `arith.extf` first), and the output is FP16 anyway. The explicit conversion
states the precision instead of relying on a compiler promotion rule.

## A useful debugging order

| Symptom | First thing to inspect |
| --- | --- |
| `TODO` | Finish the kernel and remove the wrapper's explicit guard |
| `UNAVAILABLE` | Wrong Python environment, missing Triton, or no NVIDIA CUDA device |
| Compiler says a name is undefined | Use `tl.*` inside the kernel and check argument names |
| Correct first block, wrong later blocks | Include the program ID in the address |
| Only ragged widths fail | Both load/store masks and the reduction padding identity |
| Only FP16 fails | Convert loaded values to FP32 before reducing |
| Rows repeat or overlap | Row stride and output ownership |
| Shape/dtype/device mismatch | The wrapper allocated the wrong output tensor |

Use `--verbose` for a traceback. Because GPU work is asynchronous, the reported
Python line can follow the actual bad access. Setting `CUDA_LAUNCH_BLOCKING=1`
(PowerShell: `$env:CUDA_LAUNCH_BLOCKING = "1"`) can make debugging attribution
easier. Start a fresh checker process after an illegal device memory access. This
is a diagnostic workflow, not a timing setup.

## Compare your code with the portfolio

The small exercise wrappers assume the documented inputs supplied by the checker.
The main [ops.py](../../src/kernel_portfolio/ops.py) adds explicit validation and
rejects unsupported layouts, gradients, devices and oversized index ranges. Read
one of those checks and write a test input that would trigger it.

Finally, read [cuda/kernels.cu](../../cuda/kernels.cu). Match the vector index
`blockIdx.x * blockDim.x + threadIdx.x` with the addresses produced by your
Triton program. In the CUDA softmax, find the eight shared partial sums, the
warp reduction and the barriers. Those are implementation details the Triton
compiler largely manages for your block-level description.

**Exit check:** explain your own three kernels without reading the solution.
Then choose one controlled experiment from [the optimization labs](../../docs/LABS.md).
Reference kernel patterns: [official vector-add tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
and [official softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html).
