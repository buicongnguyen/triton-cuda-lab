# Intermediate and advanced extension

> **Project record, not a learning step.** This page documents how the workshops were planned and reviewed.
> You can skip it while learning; start from the
> [README](../../README.md#the-path-step-by-step) instead.

This extension follows the completed beginner course. The deliverable is six
workshops with editable starters, separate reference implementations, independent
checks, and reproducible GPU measurements. It adds no claims of learner completion.

| Level | Workshop | Concrete implementation and acceptance |
| --- | --- | --- |
| Intermediate | Strided softmax | Address both strides; accept transposed, sliced and broadcast read-only inputs; compare to FP64 softmax |
| Intermediate | Softmax backward | Compute the vector-Jacobian product without a dense Jacobian; compare to autograd and finite differences |
| Intermediate | Fused GEMM | FP16/BF16 dot products, FP32 bias/ReLU epilogue, grouped tile scheduling; test tails and zero K |
| Advanced | Online normalization | CPU mergeable (maximum, exponential sum) state; verify partition and merge-order invariance |
| Advanced | Split softmax | Three launches for partial statistics, merged statistics and normalization; support widths through 131072 |
| Advanced | Streaming attention | Tile QK and PV with online rescaling, no full score allocation, causal/noncausal self-attention |

## Logic review before implementation

- Derive the formulas before optimizing. Tiny worked examples must expose the
  rescaling factor, the backward reduction and tail ownership.
- A negative-infinity empty maximum needs an explicit identity on the CPU.
  GPU tiles always contain at least one valid element for nonempty inputs.
- Wide softmax must merge **rescaled** partial denominators. A kernel boundary
  on the same CUDA stream provides global ordering; there is no cross-block spin barrier.
- Attention must rescale the numerator accumulator as well as its denominator.
  Real causal rows include their diagonal, avoiding fully masked rows.
- Fused GEMM rounds only after the FP32 epilogue. Its baseline must use the same
  precision contract; ordinary half-precision `A @ B + bias` has an earlier rounding.
- Attention is a bounded teaching implementation: one head, equal Q/K/V lengths,
  no dropout or backward. It is not a replacement for PyTorch's optimized SDPA.
- Preserve the beginner runner and existing kernels so saved evidence remains valid.

## Code review and execution gates

1. Keep CPU math and discovery free of GPU imports; preserve TODO and unavailable
   outcomes. Testing references never overwrites exercises.
2. Validate shapes, layouts, dtypes, devices, index spans and launch knobs before
   invoking kernels. Inputs are read-only; outputs are separately allocated.
3. Check ragged dimensions, extreme finite values, no input mutation, non-default
   streams, option variants and invalid calls. Use independent double-precision
   references and autograd, not the solution's own helper formulas alone.
4. Compile and execute all reference kernels on the available NVIDIA GPU. Run
   existing regressions and CPU-only checker tests, plus lint and local link checks.
5. Correctness-gate every benchmark variant, save raw timing samples, runtime
   metadata, shape/layout and source hashes. Record graph and event timing as
   separate experiments; retain slow results. Do not infer hardware counters.
6. Write the final self-review and measured limitations into the validation report.

Primary algorithm references: [Triton softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html),
[Triton GEMM](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html),
[online normalizer paper](https://arxiv.org/abs/1805.02867), and
[Triton attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).
The workshops are original compact teaching implementations. Public book previews
remain supplementary; unavailable book text is not assumed.

## Execution outcome

All six workshops now have lessons, starters and separate working solutions.
The workshop checker passed 11 CPU and 120 GPU checks; the complete learning
runner suite passed 14 tests. Original portfolio regressions passed with the
existing two-GPU skip. Both benchmark timing modes completed and retain all
configuration results, including losses. See the [review and validation](../../results/workshops/VALIDATION.md)
and [course entry point](../../learning/workshops/README.md).
