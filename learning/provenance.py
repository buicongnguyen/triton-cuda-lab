"""Source identity rules shared by the benchmark and CPU-only validation."""


def learner_file(name):
    """Editable student work is not part of the maintained course manifest."""
    return name.startswith(("learning/exercises/", "learning/workshops/exercises/")) or name in {
        "learning/JOURNAL.md",
        "learning/workshops/EXPERIMENTS.md",
    }


def workshop_sources(operations, *, solution):
    """Selected implementation plus shared local dependencies, relative to repo root.

    Keep this list aligned with workshop benchmark imports. It is deliberately
    explicit: unrelated exercises and unused reference answers are not measured.
    """
    folder = "solutions" if solution else "exercises"
    shared = {
        "learning/__init__.py",
        "learning/provenance.py",
        "learning/check.py",
        "learning/checks.py",
        "learning/gpu_checks.py",
        "learning/workshops/__init__.py",
        "learning/workshops/benchmark.py",
        "learning/workshops/check.py",
        "learning/workshops/checks.py",
        "learning/workshops/contracts.py",
        "learning/workshops/gpu_checks.py",
    }
    shared.update(
        f"src/kernel_portfolio/{name}.py"
        for name in (
            "__init__",
            "benchmark",
            "contracts",
            "environment",
            "ops",
            "references",
            "triton_kernels",
        )
    )
    return sorted(shared | {f"learning/workshops/{folder}/{name}.py" for name in operations})


def measured_workshop_hashes(report):
    """Select executed sources from old broad manifests; new reports must be complete.

    Version-1 reports retain their original hashes for auditability. They predate
    the dependency list and omitted some helpers, so only their recorded hashes
    can be verified. Version 2 requires exactly the declared dependencies.
    """
    operations = {case["op"] for case in report["cases"]}
    operations.update(item["op"] for item in report.get("skipped", []))
    expected = set(workshop_sources(operations, solution=report["settings"]["solution"]))
    recorded = report["source_sha256"]
    if report.get("schema_version", 1) >= 2 and set(recorded) != expected:
        raise ValueError("Workshop source manifest does not match the selected implementations")
    return {name: digest for name, digest in recorded.items() if name in expected}
