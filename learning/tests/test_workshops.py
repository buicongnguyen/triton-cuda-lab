"""Workshop runner behavior without requiring any third-party packages."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from learning.workshops import check


class WorkshopRunnerTests(unittest.TestCase):
    def run_quiet(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return check.main(args)

    def test_cpu_reference_and_level_selection(self):
        self.assertEqual(self.run_quiet(["all", "--solution"]), 0)
        self.assertEqual(self.run_quiet(["all", "--level", "advanced", "--solution"]), 0)

    def test_empty_selection_is_not_completion(self):
        self.assertEqual(self.run_quiet(["all", "--level", "intermediate"]), 3)

    def test_gpu_unavailable_is_not_completion(self):
        with patch.object(check, "gpu_available", return_value="no test GPU"):
            self.assertEqual(self.run_quiet(["streaming_attention", "--solution"]), 3)

    def test_hint_does_not_load_gpu_or_exercise(self):
        with patch.object(check, "load_exercise", side_effect=AssertionError("loaded")):
            with patch.object(check, "gpu_available", side_effect=AssertionError("GPU probed")):
                self.assertEqual(self.run_quiet(["streaming_attention", "--hint"]), 0)

    def test_starter_and_wrong_answer_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "exercises").mkdir()
            p = root / "exercises/online_normalizer.py"
            with patch.object(check, "ROOT", root):
                stubs = (
                    "def merge(left, right):\n    raise NotImplementedError('TODO')\n"
                    "def streaming_softmax(values, size):\n    raise NotImplementedError('TODO')\n"
                )
                todo = "def summarize(values):\n    raise NotImplementedError('TODO')\n"
                p.write_text(todo + stubs)
                self.assertEqual(self.run_quiet(["online_normalizer"]), 2)
                p.write_text("def summarize(values):\n    return (0, 123)\n" + stubs)
                self.assertEqual(self.run_quiet(["online_normalizer"]), 1)

    def test_syntax_error_is_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "exercises").mkdir()
            (root / "exercises/online_normalizer.py").write_text("def not valid syntax:\n")
            with patch.object(check, "ROOT", root):
                self.assertEqual(self.run_quiet(["online_normalizer"]), 1)


if __name__ == "__main__":
    unittest.main()
