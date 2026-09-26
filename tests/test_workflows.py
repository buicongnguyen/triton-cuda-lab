"""Regression checks for safe site builds and editable learner work; CPU only."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_docs

from learning.provenance import measured_workshop_hashes, workshop_sources
from scripts import build_site


class SiteBuildTests(unittest.TestCase):
    def test_unrelated_mkdocs_sources_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "source-site"
            out.mkdir()
            files = {"mkdocs.yml": "site_name: My sources\n", "notes.md": "Keep these notes\n"}
            for name, content in files.items():
                (out / name).write_text(content, encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                build_site.main(["--out", str(out)])
            self.assertEqual(exc.exception.code, 2)
            self.assertEqual({p.name: p.read_text() for p in out.iterdir()}, files)

    def test_wrong_marker_is_not_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            marker = out / build_site.BUILD_MARKER
            marker.write_text("unrelated content", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                build_site.main(["--out", str(out)])
            self.assertEqual(marker.read_text(), "unrelated content")

    def test_generated_build_can_be_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            source = root / "README.md"
            source.write_text("# First version\n", encoding="utf-8")
            out = root / "build"
            with patch.object(build_site, "ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                build_site.main(["--out", str(out)])
                (out / "obsolete.html").write_text("obsolete", encoding="utf-8")
                source.write_text("# Second version\n", encoding="utf-8")
                build_site.main(["--out", str(out)])
            self.assertEqual((out / "docs/index.md").read_text(), source.read_text())
            self.assertEqual((out / build_site.BUILD_MARKER).read_text(), build_site.BUILD_ID)
            self.assertFalse((out / "obsolete.html").exists())

    def test_repo_root_is_rejected_even_with_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / build_site.BUILD_MARKER
            marker.write_text(build_site.BUILD_ID, encoding="utf-8")
            with patch.object(build_site, "ROOT", root), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    build_site.main(["--out", str(root)])
            self.assertTrue(marker.exists())


class ProvenanceWorkflowTests(unittest.TestCase):
    def test_new_reports_select_the_requested_implementation(self):
        for solution, chosen, unused in (
            (True, "solutions", "exercises"),
            (False, "exercises", "solutions"),
        ):
            with self.subTest(solution=solution):
                paths = workshop_sources(["strided_softmax"], solution=solution)
                self.assertIn(f"learning/workshops/{chosen}/strided_softmax.py", paths)
                self.assertFalse(any(f"/{unused}/" in path for path in paths))
                self.assertFalse(any("online_normalizer" in path for path in paths))
                report = {
                    "schema_version": 2,
                    "settings": {"solution": solution},
                    "cases": [{"op": "strided_softmax"}],
                    "source_sha256": dict.fromkeys(paths, "digest"),
                }
                self.assertEqual(measured_workshop_hashes(report), report["source_sha256"])
                del report["source_sha256"][f"learning/workshops/{chosen}/strided_softmax.py"]
                with self.assertRaises(ValueError):
                    measured_workshop_hashes(report)

    def test_learner_edits_leave_reference_checks_green_but_kernel_edits_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(name, text):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                return hashlib.sha256(text.encode()).hexdigest()

            answer = "learning/workshops/solutions/strided_softmax.py"
            student = "learning/workshops/exercises/online_normalizer.py"
            beginner = "learning/exercises/offsets.py"
            helper = "learning/check.py"
            digests = {answer: write(answer, "correct answer\n"), student: write(student, "TODO\n")}
            write(
                "results/workshops/example.json",
                json.dumps(
                    {
                        "schema_version": 1,
                        "implementation": "reference solutions",
                        "settings": {"solution": True},
                        "cases": [{"op": "strided_softmax"}],
                        "source_sha256": digests,
                    }
                ),
            )
            write(
                "results/learning/source-sha256.json",
                json.dumps({helper: write(helper, "checker\n")}),
            )
            write(student, "completed CPU workshop\n")
            write(beginner, "completed beginner exercise\n")
            write("learning/JOURNAL.md", "My experiment notes\n")

            def run_checks():
                suite = unittest.TestSuite(
                    [
                        test_docs.ResultProvenanceTests(
                            "test_results_were_measured_on_current_sources"
                        ),
                        test_docs.ResultProvenanceTests(
                            "test_course_manifest_matches_course_files"
                        ),
                    ]
                )
                with patch.object(test_docs, "ROOT", root):
                    return unittest.TextTestRunner(stream=io.StringIO()).run(suite)

            result = run_checks()
            self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
            write(answer, "changed measured kernel\n")
            self.assertFalse(run_checks().wasSuccessful())

    def test_saved_source_snapshot_preserves_original_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = "learning/workshops/benchmark.py"
            snapshot = "results/workshops/source_snapshots/benchmark.py"
            (root / source).parent.mkdir(parents=True)
            (root / source).write_text("new metadata code\n", encoding="utf-8")
            (root / snapshot).parent.mkdir(parents=True)
            (root / snapshot).write_text("original benchmark\n", encoding="utf-8")
            report = {
                "implementation": "reference solutions",
                "settings": {"solution": True},
                "cases": [{"op": "strided_softmax"}],
                "source_sha256": {source: test_docs.sha256(root / snapshot)},
                "source_snapshots": {source: snapshot},
            }
            (root / "results/workshops/example.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            with patch.object(test_docs, "ROOT", root):
                test_docs.ResultProvenanceTests(
                    "test_results_were_measured_on_current_sources"
                ).debug()
            (root / snapshot).write_text("tampered snapshot\n", encoding="utf-8")
            with patch.object(test_docs, "ROOT", root), self.assertRaises(AssertionError):
                test_docs.ResultProvenanceTests(
                    "test_results_were_measured_on_current_sources"
                ).debug()


if __name__ == "__main__":
    unittest.main()
