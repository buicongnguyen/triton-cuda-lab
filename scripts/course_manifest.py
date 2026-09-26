"""Refresh results/learning/source-sha256.json, the identity of the beginner course files.

Run after editing a maintained lesson, solution or checker. tests/test_docs.py fails
while the manifest and the files disagree. The file list itself stays as recorded;
add a new course file to the JSON by hand. Learner exercises and journals are excluded.
Hashes read CRLF as LF, like the
benchmark reports, so Windows and Linux checkouts agree.

    python scripts/course_manifest.py
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.provenance import learner_file  # noqa: E402

MANIFEST = ROOT / "results/learning/source-sha256.json"


def file_sha256(path):
    # Same rule as kernel_portfolio.benchmark.file_sha256 (which imports torch).
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main():
    names = {
        name: digest
        for name, digest in json.loads(MANIFEST.read_text(encoding="utf-8")).items()
        if not learner_file(name)
    }
    missing = [name for name in names if not (ROOT / name).exists()]
    if missing:
        raise SystemExit(f"Listed files no longer exist: {', '.join(missing)}")
    fresh = {name: file_sha256(ROOT / name) for name in names}
    changed = [name for name in names if names[name] != fresh[name]]
    MANIFEST.write_text(json.dumps(fresh, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(fresh)} files; updated: {', '.join(changed) or 'none'}")


if __name__ == "__main__":
    main()
