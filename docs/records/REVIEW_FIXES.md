# Follow-up review fixes, 2026-09-26

The review of `ea1a53e` reproduced two workflow defects. These fixes preserve the
kernel implementations and existing performance measurements.

## Site output ownership

The site builder previously accepted any nonempty folder containing `mkdocs.yml`
as a previous generated build, then recursively deleted it. That also described
an ordinary MkDocs source project.

The builder now creates and checks `.kernel-lab-site-build` with an exact versioned
identifier before deleting a nonempty output directory. Existing unmarked builds
must be replaced using a new empty output directory. Tests exercise unrelated
MkDocs sources, invalid markers, rebuilds, and rejection of the repository root.
The deployment job runs only for the public repository; the private source copy
can still build and validate documentation.

## Editable coursework and measurement provenance

Reference workshop reports used to fingerprint every exercise, including a CPU
starter that the GPU benchmark never executed. Completing that exercise caused
the main validation suite to demand new GPU measurements.

New workshop reports use schema version 2 and fingerprint the selected
implementation with its shared dependencies. Validation interprets older broad
manifests according to the implementation actually measured. The maintained
course manifest also excludes editable starters and journals.

Old workshop reports keep their original hashes, timestamps, samples, and timings.
Their `source_snapshots` field points to the unchanged original runner, because
the current runner has different metadata code. Measured kernels still have to
match current sources. Tests verify that learner edits pass, measured-kernel
edits fail, missing required dependencies fail, and modified snapshots fail.

## Forward-only custom ops

The plain functions reject gradient-tracking inputs at the call. PyTorch's custom
dispatcher can accept those inputs but raises on backward because these operators
have no autograd formula. The setup guide now explains the distinction and shows
inference under `torch.no_grad()`. A GPU regression check covers the backward
error and inference behavior for all five custom ops.

## Validation

- Main suite on an RTX 4080 SUPER: 44 tests run, no failures, two expected skips
  (a second GPU is unavailable; interpreter mode runs separately).
- Checker regression suite: 15 tests passed.
- Ruff lint and formatting checks passed for 96 Python files.
- Strict MkDocs build passed.

The preceding review separately ran the interpreter, all beginner and workshop
reference checks, and a fresh native CUDA build. This change does not alter kernel
math or claim new performance results. The follow-up tests and report are automated
validation, not an independent human sign-off.
