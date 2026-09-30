"""Benchmark-native integration boundaries. Evaluation-only SVR is isolated."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from .features import Risk
from .harness import Evidence, HarnessBackend

BENCHMARKS = {
    "Terminal-Bench-Science 0.1": 70,
    "ScienceAgentBench": 102,
    "CompChemBench": 40,
    "SciAgentArena": None,  # release contains approximately 200 tasks
    "ReplicationBench": 111,
}


@dataclass(frozen=True)
class FinalEvaluation:
    native_score: float
    scientific_violations: frozenset[str]

    @property
    def svr_indicator(self) -> int:
        return int(bool(self.scientific_violations))


class PublicRiskAdapter(ABC):
    """Maps observable diagnostics to four coordinates before selection."""
    @abstractmethod
    def risk(self, evidence: list[Evidence], remaining_budget: float) -> Risk: ...


class NativeBenchmark(ABC):
    """Install official release in a frozen container; never infer outcomes from paper tables."""
    name: str
    @abstractmethod
    def tasks(self): ...
    @abstractmethod
    def fork(self, task: Any, prefix: Any): ...
    @abstractmethod
    def backend(self) -> HarnessBackend: ...
    @abstractmethod
    def public_risk_adapter(self) -> PublicRiskAdapter: ...


class HeldOutSVR(ABC):
    """Final-artifact evaluator inaccessible to controller or feature encoder."""
    version = "svr-adapter-v1.0"
    @abstractmethod
    def evaluate(self, task: Any, final_artifact: Any, final_trace: Any) -> FinalEvaluation: ...


class TerminalBenchScience(NativeBenchmark):
    name = "Terminal-Bench-Science 0.1"
    # official executable outcome; final SVR: unresolved protocol/numerical/schema/provenance


class ScienceAgentBench(NativeBenchmark):
    name = "ScienceAgentBench"
    # official program execution/task success; final SVR: split/leakage/schema/claim consistency


class CompChemBench(NativeBenchmark):
    name = "CompChemBench"
    # native output and independent recomputation; final SVR: convergence/unit/system/tolerance


class SciAgentArena(NativeBenchmark):
    name = "SciAgentArena"
    # native stepwise and terminal metric; final SVR: unresolved protocol and claim/evidence


class ReplicationBench(NativeBenchmark):
    name = "ReplicationBench"
    # native faithfulness/correctness; final SVR: stage/provenance/claim contradiction


def assert_no_evaluator_leak(controller_inputs: dict):
    banned = {"svr", "final_artifact_evaluation", "held_out_labels", "native_answer",
              "evaluator_output", "scientific_violations"}
    found = banned.intersection(controller_inputs)
    if found:
        raise ValueError(f"evaluation-only data in controller input: {sorted(found)}")
