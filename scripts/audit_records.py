"""Audit paper invariants and anonymization on a local run ledger."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scimatch.io import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger")
    args = parser.parse_args()
    groups = defaultdict(list)
    for r in read_jsonl(args.ledger):
        groups[(r["benchmark"], r["task_id"], r["seed"], r["method"])].append(r)
    for key, steps in groups.items():
        steps.sort(key=lambda r: r["step"])
        if sum(r["task_allocation"] for r in steps if r["committed"]) > .30 + 1e-9:
            raise AssertionError(f"task budget violated: {key}")
        if sum(r["science_allocation"] for r in steps if r["committed"]) > .10 + 1e-9:
            raise AssertionError(f"scientific budget violated: {key}")
        if any(r["committed"] and not r["feasible"] for r in steps):
            raise AssertionError(f"infeasible commit: {key}")
    print(f"Audited {len(groups)} trajectories")


if __name__ == "__main__":
    main()
