"""Task-disjoint development partitions and pre-outcome candidate assignment."""
from dataclasses import dataclass
import hashlib
from collections import defaultdict
import numpy as np


@dataclass(frozen=True)
class CandidateAssignment:
    benchmark: str
    model_tier: str
    task_id: str
    seed: int
    prefix_hash: str
    risk_band: str
    levels: tuple[int, int]
    propensity: float = 0.4  # each candidate's marginal inclusion probability


def task_partition(benchmark: str, model_tier: str, task_id: str, seed: int = 3407) -> str:
    """Stable fallback assignment; use stratified_task_partitions for full catalogs."""
    key = f"{seed}|{benchmark}|{model_tier}|{task_id}".encode()
    u = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    return "Dfit" if u < .70 else ("Dcal" if u < .85 else "Dtune")


def stratified_task_partitions(tasks, seed: int = 3407):
    """Exact task-group allocation, stratified by benchmark and model tier.

    Each input item is (benchmark, model_tier, task_id). Call once on the
    development catalog and freeze the returned map before candidate sampling.
    """
    strata = defaultdict(set)
    for benchmark, tier, task_id in tasks:
        strata[(benchmark, tier)].add(task_id)
    result = {}
    for (benchmark, tier), ids in sorted(strata.items()):
        ordered = sorted(ids, key=lambda task: hashlib.sha256(
            f"{seed}|{benchmark}|{tier}|{task}".encode()).digest())
        nfit = round(.70 * len(ordered))
        ncal = round(.15 * len(ordered))
        for i, task in enumerate(ordered):
            partition = "Dfit" if i < nfit else ("Dcal" if i < nfit+ncal else "Dtune")
            result[(benchmark, tier, task)] = partition
    return result


def assign_two(benchmark, model_tier, task_id, seed, prefix_hash, risk_band,
               rng: np.random.Generator) -> CandidateAssignment:
    """Call before observing candidate outcomes; fork the same replayable prefix."""
    selected = tuple(sorted(map(int, rng.choice(5, size=2, replace=False))))
    return CandidateAssignment(benchmark, model_tier, task_id, seed,
                               prefix_hash, risk_band, selected)


def validate_partitions(rows):
    groups = {}
    for r in rows:
        key = (r["benchmark"], r["model_tier"], r["task_id"])
        value = r["partition"]
        if key in groups and groups[key] != value:
            raise ValueError(f"task crosses partitions: {key}")
        groups[key] = value
        if value == "Dtest" and r.get("used_for_fit_or_tune", False):
            raise ValueError("Dtest leakage")
