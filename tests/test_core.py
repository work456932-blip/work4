import numpy as np
import pytest
from scimatch.calibration import WilsonMap, wilson_upper
from scimatch.controller import RiskLedger, allocated_fraction
from scimatch.data import assign_two, task_partition, validate_partitions
from scimatch.features import FeatureEncoder, Capability, Risk, Prefix
from scimatch.evaluation import factorial_bootstrap, scientific_violation_rate
from scimatch.adapters import FinalEvaluation


def test_feature_layout_and_no_candidate_scaling():
    cap = Capability(.2, .4, .6, .8, 1)
    risk = Risk(.1, .2, .3, .4)
    prefix = Prefix(2, 1, 4, 3, .5, 2, .25, .7, 1)
    x = np.stack([FeatureEncoder.raw(cap, risk, k, prefix) for k in range(5)])
    assert x.shape == (5, 23)
    assert np.array_equal(x[:, 9:14], np.eye(5))
    assert np.array_equal(FeatureEncoder().fit(x).transform(x)[:, 9:14], np.eye(5))


def test_one_sided_monotonic_and_disjoint_label_math():
    p = np.linspace(.01, .5, 100)
    y = np.array([int(i % 13 == 0) for i in range(100)])
    mapping = WilsonMap.fit(p, y)
    q = mapping.transform(p)
    assert len(mapping.edges) == 9 and np.all(np.diff(q) >= 0)
    assert 0 <= wilson_upper(0, 10) <= 1


def test_pathwise_allocation_cannot_exceed_budget():
    ledger = RiskLedger()
    for t in range(16):
        task, sci = ledger.thresholds(.5, 100-t, 100)
        ledger.commit(task, sci)
    assert sum(x for x, _ in ledger.allocations) <= .30
    assert sum(x for _, x in ledger.allocations) <= .10
    assert allocated_fraction(.5, 16, 100, 100) <= .30


def test_task_disjoint_assignment_and_factorial():
    assert task_partition("b", "m", "task") == task_partition("b", "m", "task")
    assignments = [assign_two("b", "m", "task", 42, str(i), "low",
                              np.random.default_rng(i)) for i in range(10)]
    assert all(len(set(x.levels)) == 2 and x.propensity == .4 for x in assignments)
    with pytest.raises(ValueError):
        validate_partitions([dict(benchmark="b", model_tier="m", task_id="x", partition="Dfit"),
                             dict(benchmark="b", model_tier="m", task_id="x", partition="Dcal")])
    rows = [dict(benchmark="b", task_id="t", seed=42, cell=c, score=v) for c, v in
            [("S00", 30), ("S10", 32.1), ("S01", 31.5), ("S11", 35)]]
    assert np.isclose(factorial_bootstrap(rows, replicates=20)["interaction"], 1.4)


def test_svr_counts_trajectories_not_labels():
    items = [FinalEvaluation(50, frozenset({"schema", "protocol"})),
             FinalEvaluation(70, frozenset())]
    assert scientific_violation_rate(items) == 50
