"""Harness-neutral calibration suite; freeze profile before benchmark evaluation."""
from collections import defaultdict
from .features import Capability

CAPABILITY_KEYS = ("tool", "repair", "state", "science", "instruction")


def estimate_capability(neutral_trials):
    """Trials are (dimension, passed) from independent calibration tasks.

    No benchmark Dtest task label, verifier output, or benchmark reward may
    appear here. Each coordinate is an empirical success frequency [0,1].
    """
    totals = defaultdict(lambda: [0, 0])
    for trial in neutral_trials:
        key = trial["dimension"]
        if key not in CAPABILITY_KEYS:
            raise ValueError(f"unknown capability dimension: {key}")
        if trial.get("benchmark_test", False) or trial.get("evaluation_only", False):
            raise ValueError("evaluation leakage into capability profile")
        totals[key][0] += int(bool(trial["passed"]))
        totals[key][1] += 1
    if any(totals[key][1] == 0 for key in CAPABILITY_KEYS):
        raise ValueError("missing neutral trials for capability coordinate")
    return Capability(*(totals[key][0]/totals[key][1] for key in CAPABILITY_KEYS))
