"""Paper metrics; all resampling keeps methods paired by benchmark and task-seed."""
from collections import defaultdict
import numpy as np


def calibration_metrics(p, y, bins=10):
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=int)
    if len(p) != len(y) or not len(p):
        raise ValueError("empty/mismatched arrays")
    p_clip = np.clip(p, 1e-12, 1-1e-12)
    order = np.argsort(p, kind="stable")
    groups = np.array_split(order, bins)
    ece = sum(len(ix)/len(p)*abs(y[ix].mean()-p[ix].mean()) for ix in groups if len(ix))
    brier = np.mean((p-y)**2)
    nll = -np.mean(y*np.log(p_clip)+(1-y)*np.log1p(-p_clip))
    return {"ece": float(ece), "brier": float(brier), "nll": float(nll)}


def one_sided_groups(base, upper, labels, groups):
    output = {}
    for name, mask in groups.items():
        ix = np.asarray(mask, dtype=bool)
        if not ix.any():
            continue
        obs, bound = np.asarray(labels)[ix].mean(), np.asarray(upper)[ix].mean()
        output[name] = {"n": int(ix.sum()), "empirical_failure": float(obs),
                        "mean_base": float(np.asarray(base)[ix].mean()),
                        "mean_upper": float(bound),
                        "upper_bound_violation": bool(obs > bound)}
    return output


def scientific_violation_rate(evaluations):
    items = list(evaluations)
    if not items:
        raise ValueError("no trajectories")
    return 100*sum(x.svr_indicator for x in items)/len(items)


def normalized_headroom(scimatch_score, h4_score):
    return (scimatch_score-h4_score)/(100-h4_score)


def paired_bootstrap(rows, method_a, method_b, replicates=10000, seed=2026):
    """Rows: benchmark, task_id, seed, method, score. Resample paired task-seed units."""
    by_benchmark = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        key = (row["task_id"], row["seed"])
        by_benchmark[row["benchmark"]][key][row["method"]] = row["score"]
    arrays = []
    for groups in by_benchmark.values():
        paired = np.asarray([(d[method_a], d[method_b]) for d in groups.values()
                             if method_a in d and method_b in d], dtype=float)
        if len(paired) != len(groups):
            raise ValueError("methods not paired within a benchmark")
        arrays.append(paired[:, 0]-paired[:, 1])
    if not arrays:
        raise ValueError("no paired benchmarks")
    rng = np.random.default_rng(seed)
    boot = np.zeros(replicates)
    for arr in arrays:
        boot += arr[rng.integers(0, len(arr), (replicates, len(arr)))].mean(axis=1)/len(arrays)
    return {"difference": float(np.mean([a.mean() for a in arrays])),
            "ci95": np.percentile(boot, [2.5, 97.5]).tolist()}


def factorial_bootstrap(rows, replicates=10000, seed=2026):
    """Paired 2x2 interaction: S11 - S01 - S10 + S00."""
    by_benchmark = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        key = (row["task_id"], row["seed"])
        by_benchmark[row["benchmark"]][key][row["cell"]] = row["score"]
    arrays = []
    for groups in by_benchmark.values():
        vals = []
        for item in groups.values():
            if set(item) != {"S00", "S10", "S01", "S11"}:
                raise ValueError("incomplete paired factorial cell")
            vals.append(item["S11"]-item["S01"]-item["S10"]+item["S00"])
        arrays.append(np.asarray(vals))
    rng = np.random.default_rng(seed)
    boot = np.zeros(replicates)
    for a in arrays:
        boot += a[rng.integers(len(a), size=(replicates, len(a)))].mean(axis=1)/len(arrays)
    return {"interaction": float(np.mean([a.mean() for a in arrays])),
            "ci95": np.percentile(boot, [2.5, 97.5]).tolist()}
