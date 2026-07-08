"""Print a metric delta table between two run_evaluation.py results files.

Usage: python compare_results.py evals/vietnamese/baseline/results.json \
                                 evals/vietnamese/finetuned/results.json
"""

import json
import sys


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    before = json.load(open(sys.argv[1], encoding="utf-8"))
    after = json.load(open(sys.argv[2], encoding="utf-8"))

    print(f"{'task':<18}{'metric':<14}{'before':>10}{'after':>10}{'delta':>10}")
    print("-" * 62)
    for task in sorted(set(before["tasks"]) | set(after["tasks"])):
        b = before["tasks"].get(task, {}).get("metrics", {})
        a = after["tasks"].get(task, {}).get("metrics", {})
        for metric in sorted(set(b) | set(a)):
            if metric == "n":
                continue
            bv, av = b.get(metric), a.get(metric)
            if bv is None or av is None:
                row = f"{bv if bv is not None else '—':>10}{av if av is not None else '—':>10}{'—':>10}"
            else:
                row = f"{bv:>10.2f}{av:>10.2f}{av - bv:>+10.2f}"
            print(f"{task:<18}{metric:<14}" + row)


if __name__ == "__main__":
    main()
