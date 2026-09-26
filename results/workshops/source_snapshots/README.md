# Original measurement source

`benchmark-ea1a53e.py` is copied byte for byte from commit `ea1a53e`'s
`learning/workshops/benchmark.py`. It produced the existing graph and event
workshop reports. Their original `source_sha256` values, timestamps, samples,
and timings are unchanged; the new `source_snapshots` field identifies this copy.

The current runner changes which dependencies are fingerprinted and uses report
schema version 2. Keeping this snapshot lets CPU validation verify the original
reporting code without pretending to have remeasured performance. Measured
solution kernels still have to match their current source files. Unused learner
starters in old manifests are ignored by dependency-aware validation.
