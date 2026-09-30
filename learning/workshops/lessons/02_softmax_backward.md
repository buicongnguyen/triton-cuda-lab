# I2. Softmax backward without a dense Jacobian

**Prerequisite:** stable softmax and the chain rule. **Time:** 2-3 sessions.
**Edit:** [softmax_backward.py](../exercises/softmax_backward.py).
**Goal:** turn a derivative into one row reduction and pointwise arithmetic.

Let `y=softmax(x)`, and let `g[i]=dL/dy[i]` be an upstream gradient. The Jacobian is

```text
dy[i]/dx[j] = y[i] * (delta(i,j) - y[j])
dx[j] = sum_i(g[i] * dy[i]/dx[j])
      = y[j] * (g[j] - sum_i(g[i]*y[i]))
```

`delta(i,j)` is one when i=j and zero otherwise. The full N-by-N Jacobian is
unnecessary. You need only `dot=sum(y*g)` per row, then `dx=y*(g-dot)`.

For `y=[0.25,0.75]` and `g=[2,-1]`, dot is `0.5-0.75=-0.25`. Therefore
`dx=[0.25*2.25, 0.75*(-0.75)]=[0.5625,-0.5625]`. Their sum is zero. This follows
from softmax being unchanged by adding the same constant to every logit.

## Implement and validate

1. Load y and g with their **own** row and column strides. Pad both with zero.
2. Convert both to FP32 before multiplying; reduce their product once. (Triton
   promotes FP16 reductions to FP32 on its own, so the checker cannot detect a
   missing conversion; write it to make the precision explicit.)
3. Store `y*(g-dot)` in a fresh contiguous output, with the original dtype.
4. Require equal shape/dtype/device and reject differentiable inputs. This is an
   explicit VJP function, not a PyTorch `autograd.Function` or training integration.

```bash
python -m learning.workshops.check softmax_backward
python -m learning.workshops.benchmark --op softmax_backward --output results/local/i2.json
```

The checker compares a float32 result against PyTorch's FP64 autograd and the
central difference of `L=sum(softmax(x)*g)` at one logit:

```text
dL/dx[j] approximately (L(x+h*e[j]) - L(x-h*e[j])) / (2*h)
h = 1e-5, with L evaluated in FP64
```

It also checks the row-sum invariant and strided gradients. These checks catch
using `sum(g)` instead of `sum(y*g)`, reducing along the wrong axis and changing
the subtraction sign. Finite differences are a diagnostic, not exact arithmetic.

## Precision is part of the interface

The function consumes the y that you supply. If you round probabilities to FP16
first, they may not sum to exactly one; the resulting VJP need not have an exact
zero sum. The half-precision oracle therefore computes from the same rounded y.
The strict row-sum check uses float32 y and allows small numerical error.

For a full training operator you would need to decide whether forward saves
rounded or higher-precision y, register autograd behavior, test gradient chaining,
and define higher-order derivative support. Those are separate tasks. The main
implementation does the first three for its own softmax; after your attempt,
compare your kernel with `_softmax_bwd` in
[triton_kernels.py](../../../src/kernel_portfolio/triton_kernels.py) and its
registration in [library.py](../../../src/kernel_portfolio/library.py). It saves the
rounded output, as PyTorch does, and does not support double backward.

## Measure the fused reduction

The benchmark baseline is an explicit FP32 PyTorch VJP expression. It includes
casts and elementwise/reduction launches, but no autograd engine traversal.
Report this baseline by name; the result is not a speedup for an entire training
step. Try both graph and event timing and explain why the ratio changes.

**Done when:** derive the VJP from the Jacobian, reproduce the two-element example,
pass autograd/finite-difference checks, and report one measurement with its scope.
Next: [fuse a GEMM epilogue](03_fused_gemm.md).
