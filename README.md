# SciMatch (anonymous research implementation)

Code companion for **SciMatch: Impedance-Matched Harness Control for Reliable Scientific Agents**, based on the supplied manuscript. It contains no author name, affiliation, personal URL, private path, credentials, or fabricated benchmark outcome.

## Paper-to-code map

| Paper | Code |
| --- | --- |
| Eq. 4 committed unresolved-failure target | harness.Transition.committed_step_label; candidate JSONL committed_step_failure |
| Eq. 7 frozen five-component capability profile | capability.py, features.Capability |
| Eq. 8 four online risk coordinates | features.Risk, benchmark PublicRiskAdapter |
| H0–H4 nested intervention ladder | harness.py |
| Eq. 9, Table 21: 23 to 64 to 32 to 1 GELU MLP | predictor.py, features.py, scripts/train_controller.py |
| Eq. 10: disjoint Dcal, ten equal-frequency bins, Wilson z=1.645, monotone max | calibration.py |
| Eq. 5–6 and Table 21: residual ledgers and clipped allocator | controller.py |
| Eq. 11: candidates, cost mask, hysteresis, recovery decision | controller.py, runner.py |
| Development sampling, task-disjoint 70/15/15 split | data.py |
| Final-artifact evaluation-only SVR | adapters.HeldOutSVR, evaluation.py |
| Paired bootstrap, factorial interaction, calibration diagnostics | evaluation.py |
| Cost separation and price multipliers | costs.py |

## Data contract

Collect replayable development prefixes. **Before observing outcomes**, sample two distinct H0–H4 candidates uniformly without replacement (marginal propensity 2/5), fork the same prefix, run the local cycle, then label whether a task-relevant failure introduced at that transition remains unresolved at commit. Do not use continuation-level final failure as the local label. Keep every prefix from a task in the same Dfit/Dcal/Dtune partition. Dtest is the official held-out benchmark evaluation set and must not enter the training JSONL.

Each JSONL candidate row includes benchmark, model_tier, task_id, seed, prefix_hash, risk_band, level, assignment_probability, partition, features (23 raw values from FeatureEncoder.raw), and committed_step_failure (0/1). A frozen run manifest records exact model revisions, environment digests, evaluator versions and pricing. Fit the predictor and disjoint calibration map with:

    python scripts/train_controller.py --candidate-jsonl data/development_candidates.jsonl --manifest config/frozen_run_manifest.json --output controller.pt

The example manifest deliberately has missing fields and is not a reproduction manifest. The audit script checks budget and commit invariants against a genuine per-step ledger:

    python scripts/audit_records.py path/to/ledger.jsonl

## Native integrations and limits of supplied material

NativeBenchmark and HarnessBackend define integration boundaries for official Terminal-Bench-Science 0.1, ScienceAgentBench, CompChemBench, SciAgentArena and ReplicationBench releases. An implementer must connect each one to a pinned official environment and verifier. HeldOutSVR is a separate final-artifact evaluator called after the controller ends; its benchmark-specific predicates are described in Appendix Table S1, but executable evaluator code is not included in the manuscript.

The paper specifies a committed-task-failure MLP and a scientific-risk ledger, but does not specify a separately fitted candidate-wise scientific-failure model. The controller requires a calibrated candidate-wise science_bound hook for dual-risk operation. The explicit task_only=True option only inspects Eq. 11's task constraint and must never be reported as dual-risk SciMatch. The paper also does not supply trained weights, neutral calibration trials, benchmark adapter code, evaluator implementation, exact release/container hashes, or the frozen run manifest. No published score is claimed or reconstructible from this source alone.

Use the native benchmark release and real observations to implement the missing integrations. Do not invent results from the manuscript tables. Tests check feature layout, Wilson calibration, pathwise accounting, partitioning, factorial arithmetic and trajectory-level SVR:

    python -m pytest tests
