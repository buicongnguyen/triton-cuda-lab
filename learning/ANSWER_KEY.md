# Answer explanations

Attempt the questions before reading. Passing `--solution` validates these
reference files, not your own completion. The complete code is in `solutions/`.

## Lesson 0

1. Output 5 of vector add does not depend on output 4; both read their own inputs.
2. Pointer arguments refer to device memory holding tensor elements. The wrapper
   supplies tensor metadata separately; the kernel does not receive a Python list.
3. Tiny CPU work can finish before GPU submission and data movement overhead is paid.
4. Allocation, validation, grid selection and launch submission remain on the host.

## Lesson 1

N=17, BLOCK=8 needs three programs. They describe offsets 0–7, 8–15 and 16–23.
Only offset 16 is valid in the last program. Each program has a disjoint interval,
so every valid output has exactly one owner. `(N+BLOCK-1)//BLOCK` rounds up without
using floating-point division; N=0 correctly returns zero.

## Lesson 2

Shape (3,2), strides (6,2), base 1 produces rows of indices `[1,3]`, `[7,9]`,
`[13,15]`. The second-row base is seven, even though there are only two logical
columns. A contiguous copy changes both the memory layout and the work measured:
it can add allocation and a data-copy kernel before the operator itself.

## Lesson 3

`[2,-1,4]` pads to `[2,-1,4,0]`, then becomes `[1,4]`, then `[5]`. Sum padding
is zero; max padding is negative infinity. The mean of the original row is 5/3,
not 5/4. Reduction trees change the order of floating-point operations, so small
rounding differences do not automatically indicate an indexing bug.

## Lesson 4

For any constant c, `exp(x_i-c)=exp(x_i)*exp(-c)`. The common positive factor
cancels from numerator and denominator. Choosing c=max(x) keeps all exponential
arguments nonpositive. Exponentials alone do not normalize the row. Zero padding
adds an extra positive `exp(0-max)` term to the denominator, even if it does not
alter the maximum. Negative-infinity padding contributes zero instead.

## Lesson 5

The residual sum and squares are per-element; mean square and inverse RMS are
per-row; learned weights are per-column. If every residual sum is zero, mean
square is zero and inverse RMS is finite because eps is positive; multiplying
by zero yields zero. FP16-rounded residuals and FP32 residuals are different
inputs to the subsequent square and reduction, so the reference must match the choice.

## Lesson 6

For M=5, N=7, BM=2, BN=3, the output grid is `ceil(5/2)*ceil(7/3)=3*3=9`
programs. With K=5 and BK=2, each program takes three K steps. The last output
row tile has one valid row, the last column tile one valid column, and the last
K tile one valid reduction element. A and B need masks on their respective
output dimension and K; the C store needs both output-dimension masks.

## Lesson 7

A ratio below one means the candidate is slower. Decimal GB/s requires both
milliseconds-to-seconds and bytes-to-gigabytes conversion: `bytes/(ms*1e6)`.
A captured graph and an ordinary Python dispatch benchmark differ in submission
overhead; neither number alone represents an entire model or service.

## Lesson 8

- **Add:** the grid owns disjoint vector chunks; invalid tail lanes neither read
  nor write outside the tensor.
- **Row sum:** one program owns one row; its masked vector reduces to one FP32
  scalar. The input's physical row stride locates the next row.
- **Softmax:** two reductions operate on the same row; padding is algebraically
  neutral; output addresses use the new contiguous layout.

A learner should also explain why no kernel computes a backward gradient here,
why `torch.empty` requires every valid output to be written, and why a tested
contract is narrower than “works for any tensor.”

## Lesson 9

256 threads / 32 threads per warp gives eight warp partials. Distinct output
ownership does not remove the need to coordinate the *intermediate reduction*.
Neighboring thread IDs load neighboring columns on each loop iteration. A
128-thread version has four warps, so the reduction must read four valid partials
rather than eight, while all threads still participate in barriers. In this code
that means setting `kThreads = 128`: `kWarps`, the scratch array and the
`lane < kWarps` read all follow from it.
