"""Do better thresholds exist? Fit the plan rules on one grid log and score them on the other.

scripts/row_plan_grid.py times every plan on a grid of shapes. This script asks whether a
different threshold (where to start splitting, up to which width to use 4096-wide chunks)
would have done better, without fooling itself: it picks the best thresholds on one run of
the grid and reports how they score on the other run. A threshold that helps only the run
it was fitted on has learned that run's noise. The current rule is the reference.

    python scripts/row_plan_fit.py results/row-plan-grid.log results/row-plan-grid-repeat.log \
        > results/row-plan-fit.log
"""

import argparse
import itertools
import statistics
from pathlib import Path

SPLIT_ROWS = (0, 8, 16, 24, 32, 40, 48, 64, 80, 100, 120, 160)  # split below this many rows
SPLIT_MIN_WIDTH = (8193, 12289, 16385, 24577, 28673, 32001, 50000, 65537)
NARROW_WIDTH = (16384, 28672, 32000)  # 4096-wide chunks up to this width ...
NARROW_ROWS = (0, 80)  # ... from this many rows
SPLIT_ABOVE = {"softmax": 8192, "rmsnorm": 16384}


def load(path):
    """(SM count, {(op, rows, width): {plan label: microseconds}}) from a row_plan_grid log."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    sms = int(lines[0].split()[0])
    names = lines[2].split()[3:9]
    cells = {}
    for line in lines[3:]:
        parts = line.split()
        if len(parts) == 11 and parts[0] in SPLIT_ABOVE:
            cells[(parts[0], int(parts[1]), int(parts[2]))] = dict(
                zip(names, map(float, parts[3:9]))
            )
    return sms, cells


def plan(op, rows, width, params, sms):
    split_rows, split_min_width, narrow_width, narrow_rows = params
    if width > max(SPLIT_ABOVE[op], split_min_width - 1) and rows < split_rows:
        return "S4096/8" if width > 65536 else "S2048/4"
    if width <= narrow_width and rows >= narrow_rows:
        return "L4096/8"
    return "L8192/16"


def regret(cells, op, params, sms):
    values = [
        times[plan(op, rows, width, params, sms)] / min(times.values()) - 1
        for (cell_op, rows, width), times in cells.items()
        if cell_op == op
    ]
    return statistics.fmean(values)


def describe(params, sms):
    split_rows, split_min_width, narrow_width, narrow_rows = params
    split = f"split below {split_rows} rows from width {split_min_width}"
    narrow = f"4096-wide chunks up to width {narrow_width} from {narrow_rows} rows"
    return f"{split}; {narrow}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    (sms, first), (sms_second, second) = load(args.first), load(args.second)
    if sms != sms_second:
        parser.error("the two logs come from GPUs with different SM counts")
    current = {
        op: (sms, SPLIT_ABOVE[op] + 1, 28672, sms) for op in SPLIT_ABOVE
    }  # the rule in triton_kernels._row_plan
    space = list(itertools.product(SPLIT_ROWS, SPLIT_MIN_WIDTH, NARROW_WIDTH, NARROW_ROWS))
    print(f"Mean regret per operator over {len(first)} cells per log; {sms} SMs; lower is better.")
    print("Regret is a plan's time over the best plan's time in the same cell, minus 1.")
    print(f"Each fit tries {len(space)} threshold combinations on one log and scores the winner")
    print("on the other log. 'gain' is the current rule's regret minus the fitted rule's, in")
    print("percentage points, on the log the fit did not see.")
    for op in SPLIT_ABOVE:
        print()
        print(op)
        now = [regret(cells, op, current[op], sms) for cells in (first, second)]
        print(f"  current rule: {100 * now[0]:.1f}% on the first log,", end=" ")
        print(f"{100 * now[1]:.1f}% on the second")
        for name, train, test, held in (("first", first, second, 1), ("second", second, first, 0)):
            best = min(space, key=lambda params: regret(train, op, params, sms))
            fitted_train = regret(train, op, best, sms)
            fitted_test = regret(test, op, best, sms)
            gain = 100 * (now[held] - fitted_test)
            print(f"  fitted on the {name} log: {describe(best, sms)}")
            print(
                f"    {100 * fitted_train:.1f}% on the log it was fitted on, "
                f"{100 * fitted_test:.1f}% on the other: gain {gain:+.1f} points"
            )


if __name__ == "__main__":
    main()
