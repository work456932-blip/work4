"""Fit on Dfit only; calibrate on Dcal only. JSONL rows have 23 raw features."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scimatch.data import validate_partitions
from scimatch.features import FeatureEncoder
from scimatch.predictor import fit_with_validation, probabilities, FitConfig
from scimatch.calibration import WilsonMap
from scimatch.io import read_jsonl, save_bundle


def rows_for(rows, partition):
    selected = [r for r in rows if r["partition"] == partition]
    x = np.asarray([r["features"] for r in selected], dtype=np.float32)
    y = np.asarray([r["committed_step_failure"] for r in selected], dtype=np.float32)
    return selected, x, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--manifest", required=True, help="Frozen release/model/container metadata JSON")
    args = p.parse_args()
    rows = list(read_jsonl(args.candidate_jsonl))
    validate_partitions(rows)
    fit, xf, yf = rows_for(rows, "Dfit")
    cal, xc, yc = rows_for(rows, "Dcal")
    if not len(fit) or not len(cal) or any(r["partition"] == "Dtest" for r in rows):
        raise ValueError("training input requires Dfit and Dcal only; Dtest is excluded")
    # Task-group 10% internal validation. All prefixes/trials from a task stay together.
    tasks = sorted({(r["benchmark"], r["model_tier"], r["task_id"]) for r in fit})
    rng = np.random.default_rng(3407)
    rng.shuffle(tasks)
    held = set(tasks[:max(1, round(len(tasks)*.10))])
    is_val = np.asarray([(r["benchmark"], r["model_tier"], r["task_id"]) in held for r in fit])
    if not is_val.any() or is_val.all():
        raise ValueError("need at least two Dfit task groups")
    encoder = FeatureEncoder().fit(xf[~is_val])
    model, diagnostics = fit_with_validation(encoder.transform(xf[~is_val]), yf[~is_val],
                                             encoder.transform(xf[is_val]), yf[is_val], FitConfig())
    calibration = WilsonMap.fit(probabilities(model, encoder.transform(xc)), yc)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest["fit_diagnostics"] = diagnostics
    manifest["candidate_counts"] = {"Dfit": len(fit), "Dcal": len(cal)}
    save_bundle(args.output, model, encoder, calibration, manifest)


if __name__ == "__main__":
    main()
