# Compare reasoning after your own attempt

Use `--solution` only when you want to run the separate reference answer. Read
the named intermediate variables and explain them before copying a pattern.

| Workshop | Reference | Central idea | Why a plausible attempt fails |
| --- | --- | --- | --- |
| I1 | [strided_softmax.py](solutions/strided_softmax.py) | Loads use both input strides; stores use contiguous output width | Assuming SC=1 silently reads the wrong elements of transposes and column slices |
| I2 | [softmax_backward.py](solutions/softmax_backward.py) | One FP32 reduction of y*g, then y*(g-dot) | A reduction of g alone omits probability weighting; a dense Jacobian wastes N-squared storage |
| I3 | [fused_gemm.py](solutions/fused_gemm.py) | A grouped grid owns output tiles; the FP32 epilogue precedes conversion | Fixed group size for the last partial group loses output tiles; casting early changes semantics |
| A1 | [online_normalizer.py](solutions/online_normalizer.py) | Re-express two summaries relative to a common maximum | Raw denominator addition combines incompatible scales; empty-empty subtraction yields NaN |
| A2 | [split_softmax.py](solutions/split_softmax.py) | Partial stats, rescaled merge, reread/normalize on one stream | Local normalized probabilities cannot be concatenated into a globally normalized row |
| A3 | [streaming_attention.py](solutions/streaming_attention.py) | Carry (m,l,a) while multiplying only one score tile at a time | Rescaling l without a corrupts every output after an increased running maximum |

## Worked answers worth memorizing by derivation

- **I1:** a 2x3 contiguous matrix transposed has strides `(1,3)`. Logical `(2,1)`
  reads storage index `2*1+1*3=5`. It is unrelated to output offset `2*2+1=5`
  except by coincidence for this final element; `(0,1)` reads 3 and writes 1.
- **I2:** y=`[.25,.75]`, g=`[2,-1]` gives dot=-.25 and
  dx=`[.5625,-.5625]`. A constant upstream gradient gives approximately zero when
  y is normalized because softmax probabilities always sum to one.
- **I3:** with 3 M-tiles, 2 N-tiles and GROUP=2, PIDs 4/5 own `(2,0)`/`(2,1)`.
  For K=0, apply bias/ReLU to the zero accumulator. For M=0 or N=0, return empty.
- **A1:** merging `(2,2)` and `(4,1)` yields `(4,1+2*exp(-2))`.
  At a newer maximum, old contributions shrink instead of overflowing.
- **A2:** M=4,N=32769,C=1024 needs T=33 and `8*4*(33+1)=1088` scratch bytes.
  Same-stream launch order supplies the producer/consumer dependency.
- **A3:** logits `[0,0,log(2)]` and values `[1,3,10]` yield 6. Keeping the old
  numerator unscaled yields 7, which catches the missing-alpha bug.

## Reference tradeoffs to challenge

The kernels deliberately keep tuning spaces small. They do not autotune on every
call. GEMM uses fixed BN/BK; split softmax allocates four scratch tensors; causal
attention still computes masked future tiles. Those are study choices, not best
practices for every deployment. If you improve one, rerun correctness and capture
new timing JSON rather than replacing the original evidence silently.

The attention oracle materializes FP64 scores only in the checker/benchmark
setup. This is independent validation outside the timed candidate call. The
candidate itself allocates only its output globally; compiler-generated spills
are a separate question that requires profiling.
