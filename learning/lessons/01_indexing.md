# 1. Offsets and masks: which values belong to me?

**Time:** 45–60 minutes. **Prerequisite:** lesson 0 and integer division.
**Edit:** [offsets.py](../exercises/offsets.py).

## Work a tiny case before writing code

Let N=10 values and BLOCK=4 values per Triton program. You need
`ceil(10/4)=3` programs, because two programs cover only eight values.

| Program ID | Local positions | Global offsets | Valid load/store mask |
| --- | --- | --- | --- |
| 0 | 0,1,2,3 | 0,1,2,3 | True,True,True,True |
| 1 | 0,1,2,3 | 4,5,6,7 | True,True,True,True |
| 2 | 0,1,2,3 | 8,9,10,11 | True,True,False,False |

Two formulas give every entry:

```text
offset = program_id * BLOCK + local_position
valid  = offset < N
```

Index N is already out of bounds: a vector of length 10 has indices 0 through 9.
The mask does not remove an offset from the block. It tells a memory operation
which lanes are allowed to access the pointer and which use a substitute value.

## Your implementation

First implement `ceil_div`. Integer division discards the fractional part, so
`10 // 4` is only 2. Think about how much to add before `//` to round upward.
Handle N=0 without creating a fictitious program.

Then implement `block_offsets`, returning both lists. Preserve their length:
the last program still has four logical positions even though only two are valid.

```bash
python -m learning.explain indexing
python -m learning.check offsets
python -m learning.check offsets --hint
```

Try N=1, N=4, N=5 and N=0 on paper. Change BLOCK to 8. For N=10, the number
of **programs** changes from 3 to 2; the number of useful output elements stays 10.

## How it appears in the real kernel

Read `_add` in [triton_kernels.py](../../src/kernel_portfolio/triton_kernels.py):

```python
offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
x = tl.load(X + offset, offset < N, other=0)
tl.store(OUT + offset, x, offset < N)
```

`X + offset` is pointer arithmetic in **elements**. Triton knows the pointer's
element type. You do not multiply by four yourself for FP32. The load substitutes
zero for masked lanes; the masked store writes nothing for those lanes.

## Common bugs and their symptoms

| Mistake | What happens | Revealing input |
| --- | --- | --- |
| `N // BLOCK` grid size | Tail outputs never get written | N=257, BLOCK=256 |
| Use `<= N` in mask | One invalid element is accessed | N=256 |
| Forget `program_id * BLOCK` | All programs write the first block | N=1024 |
| Mask the store but not the load | Reads can still go out of bounds | N=257 |

**Exit check:** draw the offsets and masks for N=17, BLOCK=8. Explain which
program owns element 16 and why each valid output has exactly one owner.
Then continue to [strides](02_strides.md).
