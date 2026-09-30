"""Residual-risk controller, distinct from the base predictor and calibration."""
from dataclasses import dataclass, field
from typing import Callable, Optional
import math
import numpy as np
from .features import Capability, Risk, Prefix, FeatureEncoder
from .calibration import WilsonMap


LEVELS = ("H0", "H1", "H2", "H3", "H4")


@dataclass(frozen=True)
class ControllerConfig:
    epsilon_task: float = 0.30
    epsilon_science: float = 0.10
    max_decisions: int = 16
    hysteresis: float = 0.02
    max_local_recovery: int = 2
    max_model_escalations: int = 1


@dataclass(frozen=True)
class Decision:
    step: int
    level: Optional[int]
    task_threshold: float
    science_threshold: float
    base: tuple[float, ...]
    upper: tuple[float, ...]
    science_upper: tuple[float, ...]
    feasible: tuple[bool, ...]
    reason: str


def allocated_fraction(science_risk: float, remaining: int, current_budget: float,
                       initial_budget: float) -> float:
    """Appendix Table 21: alpha = clip(.30-.12*r_sci-.05*(1-beta)+.03/sqrt(R),.12,.30)."""
    if remaining < 1 or initial_budget <= 0:
        raise ValueError("invalid horizon or initial budget")
    beta = max(0.0, min(1.0, current_budget / initial_budget))
    return max(0.12, min(0.30, 0.30 - 0.12*science_risk -
                           0.05*(1.0-beta) + 0.03/math.sqrt(remaining)))


@dataclass
class RiskLedger:
    config: ControllerConfig = field(default_factory=ControllerConfig)
    task_remaining: float = field(init=False)
    science_remaining: float = field(init=False)
    committed: int = 0
    allocations: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        self.task_remaining = self.config.epsilon_task
        self.science_remaining = self.config.epsilon_science

    def thresholds(self, science_risk: float, execution_budget: float,
                   initial_execution_budget: float) -> tuple[float, float]:
        remaining = max(1, self.config.max_decisions - self.committed)
        alpha = allocated_fraction(science_risk, remaining, execution_budget,
                                   initial_execution_budget)
        return alpha*self.task_remaining, alpha*self.science_remaining

    def commit(self, task_allocation: float, science_allocation: float):
        if self.committed >= self.config.max_decisions:
            raise RuntimeError("decision horizon exhausted")
        if (task_allocation < 0 or science_allocation < 0 or
            task_allocation > self.task_remaining + 1e-12 or
            science_allocation > self.science_remaining + 1e-12):
            raise ValueError("allocation exceeds residual risk")
        self.task_remaining = max(0.0, self.task_remaining-task_allocation)
        self.science_remaining = max(0.0, self.science_remaining-science_allocation)
        self.allocations.append((task_allocation, science_allocation))
        self.committed += 1
        assert sum(t for t, _ in self.allocations) <= self.config.epsilon_task + 1e-10
        assert sum(s for _, s in self.allocations) <= self.config.epsilon_science + 1e-10


class SciMatchController:
    """Predict all five counterfactual candidates before executing one.

    The paper specifies a task-failure predictor but does not specify a separate
    scientific-failure predictor. science_bound is an injected, calibrated
    candidate-wise upper-bound interface, fitted off Dtest; absent such a
    bound, use task_only=True explicitly and do not claim scientific-risk
    feasibility. A risk-sensor coordinate is not itself a probability.
    """

    def __init__(self, model, encoder: FeatureEncoder, calibration: WilsonMap,
                 costs: tuple[float, ...], initial_execution_budget: float,
                 science_bound: Optional[Callable[[Capability, Risk, Prefix, int], float]] = None,
                 config: ControllerConfig = ControllerConfig(), task_only: bool = False):
        if len(costs) != 5 or any(c <= 0 for c in costs):
            raise ValueError("five positive candidate costs required")
        if science_bound is None and not task_only:
            raise ValueError("provide calibrated candidate scientific bound or set task_only=True")
        self.model, self.encoder, self.calibration = model, encoder, calibration
        self.costs = tuple(float(c) for c in costs)
        self.initial_execution_budget = float(initial_execution_budget)
        self.science_bound = science_bound
        self.task_only = task_only
        self.config = config
        self.ledger = RiskLedger(config)
        self.previous_level: Optional[int] = None

    def decide(self, capability: Capability, risk: Risk, prefix: Prefix,
               execution_budget: float) -> Decision:
        from .predictor import probabilities
        t = self.ledger.committed + 1
        if t > self.config.max_decisions or execution_budget <= 0:
            return Decision(t, None, 0, 0, (), (), (), (), "stop")
        task_tau, science_tau = self.ledger.thresholds(
            risk.science, execution_budget, self.initial_execution_budget)
        raw = np.stack([self.encoder.raw(capability, risk, k, prefix) for k in range(5)])
        base = probabilities(self.model, self.encoder.transform(raw))
        upper = np.asarray(self.calibration.transform(base))
        science = np.asarray([self.science_bound(capability, risk, prefix, k)
                              for k in range(5)]) if self.science_bound else np.full(5, np.nan)
        feasible = tuple(bool(upper[k] <= task_tau and self.costs[k] <= execution_budget
                              and (self.task_only or science[k] <= science_tau))
                         for k in range(5))
        admissible = sorted((k for k in range(5) if feasible[k]),
                            key=lambda k: (self.costs[k], k))
        if not admissible:
            level, reason = None, "recover_or_escalate"
        else:
            level, reason = admissible[0], "least_cost_feasible"
            prev = self.previous_level
            if prev is not None and feasible[prev]:
                # Hysteresis never licenses a nominally infeasible commit.
                if self.costs[level] < self.costs[prev]:
                    if upper[level] > task_tau-self.config.hysteresis:
                        level, reason = prev, "hysteresis_hold"
                elif self.costs[level] >= self.costs[prev]:
                    level, reason = prev, "hysteresis_hold"
        return Decision(t, level, task_tau, science_tau, tuple(map(float, base)),
                        tuple(map(float, upper)), tuple(map(float, science)), feasible, reason)

    def commit(self, decision: Decision, transition_committed: bool):
        if not transition_committed:
            return  # failed local recovery or early stop: charge no allocation
        if decision.level is None or not decision.feasible[decision.level]:
            raise ValueError("cannot commit an infeasible transition")
        if decision.step != self.ledger.committed + 1:
            raise ValueError("stale decision")
        self.ledger.commit(decision.task_threshold, decision.science_threshold)
        self.previous_level = decision.level
