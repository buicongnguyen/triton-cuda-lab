# Beginner-focused improvement plan

> **Project record, not a learning step.** This page documents how the beginner course was planned and reviewed.
> You can skip it while learning; start from the
> [README](../../README.md#the-path-step-by-step) instead.

The first version demonstrated working kernels but asked the reader to make too
many jumps: from tensor notation to pointer arithmetic, from a formula to a
reduction, and from a benchmark number to an explanation. This revision follows
the user's preference for **beginner explanations and exercises**.

## Concrete changes

1. Add a guided starting page with a dependency check and a short first session.
2. Write individual lessons with prerequisites, vocabulary, tiny numerical
   examples, links to exact files, common mistakes, commands and exit criteria.
3. Provide seven CPU exercises that need only Python's standard library, followed
   by three real Triton exercises. Keep student files separate from solutions.
4. Implement a lesson checker with named cases, readable expected/actual failures,
   hints and separate outcomes for incomplete work, errors and unavailable GPUs.
5. Add execution traces for indexing, reduction, softmax and matrix tiling so the
   learner can inspect intermediate values rather than memorize finished code.
6. Add a learning journal, answer key and precise practice schedule. Connect each
   exercise to the existing measured kernels.
7. Test the checker, execute every CPU and GPU reference solution, run the existing
   portfolio regression suite, and record the review and evidence.

## Design review before implementation

- CPU lessons must not import PyTorch or Triton: a beginner can start before GPU setup.
- An unfinished starter is expected, not a failing repository regression test.
- Passing the solutions checks verifies the teaching material, not the learner's work.
- GPU checks must execute GPU code; an unavailable GPU must never appear as a pass.
- Small examples use explicit lists, indices and loops before compact tensor syntax.
- Exercise correctness is checked with concrete examples and mathematical
  invariants. No speed threshold is imposed on a first implementation.
- Reference solutions remain separate and opt-in. The checker never overwrites
  a learner's files or writes completion claims on the learner's behalf.
- Existing implementation and measurements remain available as the next stage.
  This revision deepens the learning path rather than claiming new speed records.

## Implementation and review outcome

Delivered ten individual lessons (orientation through CUDA cooperation), ten
editable exercises, ten separate solutions, five printable execution traces,
the exercise checker, answer explanations, a journal and an 18-session schedule.
The main README now starts with a 20-minute CPU-only session.

Self-review checked the following concrete issues:

| Issue | Resolution |
| --- | --- |
| Beginner setup depended on GPU packages | Seven exercises run under `python -S`; WSL CPU commands are supplied for this machine |
| Finished samples encouraged reading without practicing | Starter files are separate, unfinished, and checked by named behavioral cases |
| A missing GPU might appear as successful validation | Explicit unavailable outcome and nonzero exit code; tested by the checker suite |
| Passing references could be confused with personal progress | Clearly labeled solution mode; no progress file is auto-filled |
| Shape and stride explanations could lead to double-counted offsets | Explain backing-storage indexing versus a tensor pointer already adjusted to its first element |
| Floating-point examples could imply Python simulates FP32 | Distinguish Python float reasoning from FP32 GPU accumulation explicitly |
| Learner edits could break the checker's own regression tests | Harness tests use temporary fake starters rather than asserting real exercises stay unfinished |
| New material could invalidate old performance evidence | Original operator code and measurement hashes remain unchanged |
| Course prerequisites and schedule could disagree | GPU coding requires lessons 0–6; the measurement lesson can follow first GPU execution |

Executed evidence: 40 CPU solution checks, 34 GPU solution checks, eight checker
regression tests, and the original portfolio suite (15 pass, one two-GPU skip).
See [beginner validation](../../results/learning/VALIDATION.md). This was an assistant
self-review and local execution, not an independent human review.
