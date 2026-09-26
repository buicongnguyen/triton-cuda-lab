"""Docs and saved results must agree with the code. Standard library only.

Quoted timings must exist in the saved results, and every saved result must have been
measured on the current sources (or, in results/archive, on the archived kernel).
"""

import hashlib
import json
import re
import unittest
from pathlib import Path

from learning.provenance import learner_file, measured_workshop_hashes

ROOT = Path(__file__).resolve().parents[1]
DOCS = (
    "docs/CASE_STUDIES.md",
    "learning/lessons/07_measurement.md",
    "results/workshops/VALIDATION.md",
)
# Medians are quoted in microseconds with three decimals, as the benchmarks print them.
# A trailing "x" marks a ratio such as 1.053x, not a timing.
QUOTED_TIMING = re.compile(r"(?<![\d.])\d+\.\d{3}(?![\d.x])")


def recorded_microseconds():
    values = set()
    for path in (ROOT / "results").rglob("*.json"):
        if "local" in path.parts:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        cases = data.get("cases") if isinstance(data, dict) else None
        groups = [case["variants"] for case in cases] if cases else [data.get("variants", {})]
        for variants in groups:
            for stats in variants.values():
                if isinstance(stats, dict) and "median_ms" in stats:
                    values.add(f"{stats['median_ms'] * 1000:.3f}")
    return values


class DocNumberTests(unittest.TestCase):
    def test_quoted_timings_exist_in_results(self):
        recorded = recorded_microseconds()
        self.assertTrue(recorded, "no saved results found")
        for doc in DOCS:
            text = (ROOT / doc).read_text(encoding="utf-8")
            for number in QUOTED_TIMING.findall(text):
                with self.subTest(doc=doc, number=number):
                    self.assertIn(
                        number,
                        recorded,
                        f"{number} us in {doc} matches no saved median; update the doc",
                    )


def sha256(path):
    # Same rule as kernel_portfolio.benchmark.file_sha256: CRLF counts as LF.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


class ResultProvenanceTests(unittest.TestCase):
    def test_results_were_measured_on_current_sources(self):
        # Portfolio reports key files by name; workshop reports by repo-relative path.
        checked = 0
        for path in sorted((ROOT / "results").rglob("*.json")):
            if {"local", "archive"} & set(path.relative_to(ROOT).parts):
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            digests = report.get("source_sha256")
            if report.get("implementation") in ("reference solutions", "learner exercises"):
                digests = measured_workshop_hashes(report)
            for name, digest in (digests or {}).items():
                snapshot = report.get("source_snapshots", {}).get(name)
                source = (
                    ROOT / (snapshot or name)
                    if "/" in name
                    else ROOT / "src/kernel_portfolio" / name
                )
                with self.subTest(result=path.name, source=name):
                    self.assertTrue(source.exists(), f"{name} no longer exists")
                    self.assertEqual(
                        sha256(source), digest, f"{name} changed after {path.name}: re-measure"
                    )
                    checked += 1
        self.assertGreater(checked, 0, "no result carries source hashes")

    def test_course_manifest_matches_course_files(self):
        manifest = ROOT / "results/learning/source-sha256.json"
        for name, digest in json.loads(manifest.read_text(encoding="utf-8")).items():
            with self.subTest(source=name):
                self.assertFalse(learner_file(name), "Learner work must remain editable")
                self.assertEqual(
                    sha256(ROOT / name), digest, f"{name} changed: run scripts/course_manifest.py"
                )

    def test_archived_results_match_archived_kernels(self):
        archive = ROOT / "results/archive"
        kernels = {sha256(p) for p in archive.glob("*triton_kernels.py")}
        for path in sorted(archive.glob("*.json")):
            digests = json.loads(path.read_text(encoding="utf-8"))["source_sha256"]
            with self.subTest(result=path.name):
                self.assertIn(digests["triton_kernels.py"], kernels)


if __name__ == "__main__":
    unittest.main()
