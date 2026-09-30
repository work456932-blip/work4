"""Only observable pre-action features enter the committed-step predictor."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Capability:
    tool: float
    repair: float
    state: float
    science: float
    instruction: float

    def vector(self):
        x = np.asarray((self.tool, self.repair, self.state, self.science, self.instruction), dtype=np.float32)
        if np.any((x < 0) | (x > 1)):
            raise ValueError("capability frequencies must be in [0,1]")
        return x


@dataclass(frozen=True)
class Risk:
    execution: float
    state: float
    science: float
    budget: float

    def vector(self):
        x = np.asarray((self.execution, self.state, self.science, self.budget), dtype=np.float32)
        if not np.all(np.isfinite(x)) or np.any((x < 0) | (x > 1)):
            raise ValueError("risk coordinates must be finite values in [0,1]")
        return x


@dataclass(frozen=True)
class Prefix:
    tool_errors: int
    repeated_actions: int
    checkpoint_age: int
    unresolved_subtasks: int
    context_utilization: float
    guard_alerts: int
    verifier_history: float
    elapsed_normalized_cost: float
    remaining_execution_budget: float

    def vector(self):
        counts = (self.tool_errors, self.repeated_actions, self.checkpoint_age,
                  self.unresolved_subtasks, self.guard_alerts)
        if any(v < 0 for v in counts):
            raise ValueError("count features must be nonnegative")
        values = (math.log1p(self.tool_errors), math.log1p(self.repeated_actions),
                  math.log1p(self.checkpoint_age), math.log1p(self.unresolved_subtasks),
                  self.context_utilization, math.log1p(self.guard_alerts),
                  self.verifier_history, self.elapsed_normalized_cost,
                  self.remaining_execution_budget)
        x = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(x)):
            raise ValueError("non-finite prefix feature")
        return x


class FeatureEncoder:
    """23 = 5 capability + 4 risk + 5 candidate one-hot + 9 trajectory.

    Continuous features are standardized using Dfit statistics only. Candidate
    indicators are never standardized. Count preprocessing precedes fitting.
    """
    continuous_indices = np.array([*range(5), *range(5, 9), *range(14, 23)])

    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)

    @staticmethod
    def raw(capability: Capability, risk: Risk, level: int, prefix: Prefix) -> np.ndarray:
        if level not in range(5):
            raise ValueError("candidate level must be H0..H4")
        hot = np.eye(5, dtype=np.float32)[level]
        return np.concatenate((capability.vector(), risk.vector(), hot, prefix.vector()))

    def fit(self, raw_dfit: np.ndarray):
        x = np.asarray(raw_dfit, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 23 or len(x) == 0:
            raise ValueError("expected nonempty Dfit feature matrix with 23 columns")
        self.mean = x[:, self.continuous_indices].mean(axis=0)
        self.std = x[:, self.continuous_indices].std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("fit encoder on Dfit first")
        x = np.asarray(raw, dtype=np.float32).copy()
        if x.shape[-1] != 23:
            raise ValueError("expected 23 features")
        x[..., self.continuous_indices] = (x[..., self.continuous_indices] - self.mean) / self.std
        return x

    def state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}
