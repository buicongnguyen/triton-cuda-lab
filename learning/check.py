"""Run from the repo root: python -m learning.check offsets [--hint | --solution]."""

import argparse
import importlib.util
import traceback
from pathlib import Path

from learning.checks import CPU_EXERCISES, GPU_EXERCISES, HINTS, cpu_cases

ROOT = Path(__file__).resolve().parent


def load_exercise(path):
    spec = importlib.util.spec_from_file_location(f"exercise_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_checks(module, cases, *, verbose=False):
    """Run every case; an unfinished function marks TODO without hiding later results."""
    passed, failed, todo = 0, 0, False
    for label, check in cases:
        try:
            check(module)
        except NotImplementedError as error:
            todo = True
            print(f"  TODO {label}: {error}")
        except Exception as error:
            failed += 1
            print(f"  FAIL {label}: {type(error).__name__}: {error}")
            if verbose:
                traceback.print_exc()
        else:
            passed += 1
            print(f"  PASS {label}")
    return passed, failed, todo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exercise", choices=["all", *CPU_EXERCISES, *GPU_EXERCISES], nargs="?")
    parser.add_argument("--list", action="store_true", help="List the learning order")
    parser.add_argument("--hint", action="store_true", help="Show hints without loading your code")
    parser.add_argument(
        "--solution", action="store_true", help="Check reference files, not your work"
    )
    parser.add_argument("--include-gpu", action="store_true", help="Include GPU lessons with all")
    parser.add_argument("--verbose", action="store_true", help="Include tracebacks on failures")
    args = parser.parse_args(argv)
    if args.list or not args.exercise:
        for number, name in enumerate((*CPU_EXERCISES, *GPU_EXERCISES), start=1):
            device = "GPU + Triton" if name in GPU_EXERCISES else "Python only"
            print(f"{number:2}. {name:16} {device}")
        return 0
    names = (
        list(CPU_EXERCISES) + (list(GPU_EXERCISES) if args.include_gpu else [])
        if args.exercise == "all"
        else [args.exercise]
    )
    if args.hint:
        for name in names:
            print(f"{name}: {HINTS[name]}")
        return 0
    print("REFERENCE SOLUTIONS (not learner completion)" if args.solution else "YOUR EXERCISES")
    failed = incomplete = unavailable = 0
    passed = 0
    for name in names:
        path = ROOT / ("solutions" if args.solution else "exercises") / f"{name}.py"
        print(f"\n{name}: {path}")
        if name in GPU_EXERCISES:
            from learning.gpu_checks import gpu_available, gpu_cases

            reason = gpu_available()
            if reason:
                print(f"  UNAVAILABLE: {reason}; see docs/SETUP.md")
                unavailable += 1
                continue
            cases = gpu_cases(name)
        else:
            cases = cpu_cases(name)
        try:
            module = load_exercise(path)
        except Exception as error:
            print(f"  LOAD ERROR: {type(error).__name__}: {error}")
            failed += 1
            continue
        good, bad, todo = run_checks(module, cases, verbose=args.verbose)
        passed += good
        failed += bad
        incomplete += int(todo)
    print(
        f"\n{passed} checks passed; {failed} failed; {incomplete} exercises incomplete; "
        f"{unavailable} unavailable."
    )
    if failed:
        return 1
    if incomplete:
        return 2
    if unavailable:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
