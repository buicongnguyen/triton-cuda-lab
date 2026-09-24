# 2. Shape is not a memory address

**Time:** 45–60 minutes. **Prerequisite:** lesson 1.
**Edit:** [strides.py](../exercises/strides.py).

## A contiguous matrix

Suppose flat storage is `[0,1,2,3,4,5,6,7,8,9]`. Treat it as two rows of five:

```text
row 0:  0 1 2 3 4
row 1:  5 6 7 8 9
```

The physical distance from one row start to the next is five elements. The
distance from one column to the next is one element. These distances are strides.

```text
storage_index = storage_offset + row * row_stride + column * column_stride
```

For row 1, column 2: `0 + 1*5 + 2*1 = 7`.

## Crop each row without copying

Select the first three columns. The logical shape is now 2-by-3, but the second
row still starts at storage index 5:

```text
view row 0: 0 1 2      (3 and 4 remain in storage)
view row 1: 5 6 7      (8 and 9 remain in storage)
```

If a kernel uses `row * logical_width + column`, it will read `[3,4,5]` for
the second row. The correct row stride remains 5. No amount of tail masking can
repair an incorrect address formula for valid elements.

## Transpose and slice offsets

For storage `[0,1,2,3,4,5]` interpreted as 2-by-3, a transposed 3-by-2 view has
row stride 1 and column stride 3. Its rows are `[0,3]`, `[1,4]`, `[2,5]`.
For a slice starting at the second stored element, add `storage_offset=1`.

In PyTorch, a tensor's `data_ptr()` already refers to its first logical element.
Our Python exercise explicitly includes `storage_offset` because it indexes the
original backing list. Do **not** add PyTorch's storage offset again to an already
adjusted tensor pointer in a Triton kernel.

## Implement and inspect

Implement `offset()` first. Use it in `read_view()` with one loop per logical
dimension. The checker includes a transpose even though the portfolio's row
kernels intentionally reject noncontiguous columns.

```bash
python -m learning.explain strides
python -m learning.check strides
```

Optional: with PyTorch installed, inspect a real view. A CPU-only install is enough
for this snippet (`python -m pip install torch` in PowerShell), or use the GPU
environment:

```python
import torch

x = torch.arange(10).reshape(2, 5)
y = x[:, :3]
print(y.shape, y.stride())  # shape [2,3], strides (5,1)
print(y.is_contiguous())  # False
print(y.contiguous().stride())  # (3,1); this allocates/copies
```

Read `c.rows` in [contracts.py](../../src/kernel_portfolio/contracts.py) and
`STRIDE` in `_row_sum`. The implementation accepts row gaps while requiring
contiguous columns. This keeps the addressing and coalescing discussion manageable.

**Exit check:** a view has shape (3,2), strides (6,2), base offset 1. Find the
six storage indices. Explain why calling `.contiguous()` inside a timed wrapper
would change the benchmark's workload. Next: [reductions](03_reductions.md).
