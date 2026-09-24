"""Run intermediate/advanced work: python -m learning.workshops.check --list."""

import argparse
from pathlib import Path

from learning.check import load_exercise, run_checks
from learning.gpu_checks import gpu_available
from learning.workshops.checks import WORKSHOPS, cpu_cases

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exercise", nargs="?", choices=["all", *WORKSHOPS])
    parser.add_argument("--level", choices=["all", "intermediate", "advanced"], default="all")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--hint", action="store_true")
    parser.add_argument("--solution", action="store_true")
    parser.add_argument("--include-gpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    selected = {
        name: data
        for name, data in WORKSHOPS.items()
        if args.level == "all" or data[0] == args.level
    }
    if args.list or not args.exercise:
        for name, (level, device, hint) in selected.items():
            print(f"{name:22} {level:14} {device}")
        return 0
    if args.exercise != "all" and args.exercise not in selected:
        parser.error("That exercise is not in the selected level")
    names = (
        [name for name, data in selected.items() if args.include_gpu or data[1] == "CPU"]
        if args.exercise == "all"
        else [args.exercise]
    )
    if not names:
        print("No CPU workshops at this level. Add --include-gpu, or name a GPU workshop.")
        return 3
    if args.hint:
        for name in names:
            print(f"{name}: {WORKSHOPS[name][2]}")
        return 0
    print("REFERENCE SOLUTIONS (not learner completion)" if args.solution else "YOUR EXERCISES")
    passed = failed = incomplete = unavailable = 0
    for name in names:
        path = ROOT / ("solutions" if args.solution else "exercises") / f"{name}.py"
        print(f"\n{name}: {path}")
        if WORKSHOPS[name][1] == "GPU":
            reason = gpu_available()
            if reason:
                print(f"  UNAVAILABLE: {reason}; see docs/SETUP.md")
                unavailable += 1
                continue
            from learning.workshops.gpu_checks import gpu_cases

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
    return 1 if failed else 2 if incomplete else 3 if unavailable else 0


if __name__ == "__main__":
    raise SystemExit(main())
