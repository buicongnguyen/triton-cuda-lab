"""Timings quoted in the docs must exist in the saved results they cite. Standard library only."""

import json
import re
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
