# Beginner-path validation

Executed on 2026-09-23 and last re-executed on 2026-09-30. The
[second review](../../docs/records/REVIEW.md#second-review-2026-09-24) made kernel sizes
runtime arguments and changed how the checker reports unfinished work. The course
focuses on explanations, editable exercises and feedback; it makes no performance claims.

| Check | Observed result | Evidence |
| --- | --- | --- |
| Seven CPU reference exercises | **40 named checks passed** | [cpu-solutions.log](cpu-solutions.log) |
| Three Triton reference exercises | **36 GPU checks passed** on RTX 4080 SUPER, including two that launch the add kernel into a guarded buffer and fail if any lane stores past N | [all-solutions.log](all-solutions.log), which includes CPU and GPU checks |
| Checker regression tests | **15 tests passed** (9 checker, 6 workshop runner) | [checker-tests.log](checker-tests.log) |
| Portfolio regression suite | **48 of 49 passed; the two-GPU test skipped, and the interpreter-only class, skipped at setup, runs separately (5 passed)**; GPU tests also check that the add and fused-GEMM checkers reject deliberately broken kernels | [portfolio-regression.log](portfolio-regression.log); interpreter run in [interpreter.log](../interpreter.log) |
| Unfinished starter behavior | Reports every TODO check and returns exit code 2 | [starter-example.log](starter-example.log) |
| Static review | Ruff passed across src, tests, scripts, examples and learning | Local Ruff execution |
| Worked trace commands | Indexing, strides, reduction, softmax and matmul all ran | `python -m learning.explain TOPIC` |
| Course source identity | Manifest of the 30 maintained course files (editable exercises and the journal are excluded), refreshed by `scripts/course_manifest.py` and checked against the files by `tests/test_docs.py` | [source-sha256.json](source-sha256.json); workshop sources are identified by `source_sha256` in their reports |

CPU solutions and checker tests were executed with **`python -S`**, which disables
site-package loading. This verifies that the first seven exercises do not need
PyTorch, Triton, NumPy or a GPU. GPU reference solutions ran with the existing
WSL Python 3.12/PyTorch 2.11/Triton 3.6 environment. They exercise actual GPU
loads, reductions and stores, including ragged sizes, FP16/FP32, row gaps and
extreme finite softmax values.

The nine checker tests cover good reference answers, unfinished code, a wrong
answer, NaN comparison, hints without loading code, syntax errors, unavailable
GPU handling, separate learner/reference paths, and an unfinished function that
must not hide results for a later, finished one. They create temporary fake
student files instead of depending on real starters remaining unfinished, so
learners can freely complete their exercises without breaking the harness tests.

Reference execution is explicitly labeled **not learner completion**. The ten
starter files intentionally contain TODOs; no personal progress is inferred from
the reference checks. The minimal GPU exercise wrappers assume the checker's
stated contracts; the production-style validations remain in `src/kernel_portfolio/ops.py`.

Reproduce from the repository root:

```bash
python -S -m learning.check all --solution
python -S -m unittest discover -s learning/tests -v
# In the configured GPU environment:
python -m learning.check all --include-gpu --solution
KERNEL_REQUIRE_GPU=1 python -m unittest discover -s tests -v
```

The local GitHub workflow now includes the learning tests. It has not been run
on GitHub. Compiler, device and profiling limitations from the original
[validation record](../VALIDATION.md) remain unchanged.
