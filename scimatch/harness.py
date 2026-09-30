"""Nested intervention ladder. Backends supply actual model and environment I/O."""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol, Any


class Level(IntEnum):
    CORE = 0
    TYPED_TOOLS = 1
    STATE = 2
    GUARDS = 3
    VERIFY_RECOVER = 4


@dataclass(frozen=True)
class Evidence:
    family: str  # structural, numerical, protocol, provenance, state, execution, budget
    event: str
    severity: float
    artifact_field: str
    source: str


@dataclass
class Transition:
    observation: Any
    artifact: Any
    evidence: list[Evidence] = field(default_factory=list)
    resolved: bool = False
    introduced_failure: bool = False
    resource: dict[str, float] = field(default_factory=dict)

    @property
    def committed_step_label(self) -> int:
        """1{A_t}: new task-relevant failure still unresolved at commit."""
        return int(self.introduced_failure and not self.resolved)


class HarnessBackend(Protocol):
    def core_execute(self, task: Any, state: Any, model_id: str, decoding: dict) -> Transition: ...
    def typed_tools(self, transition: Transition) -> Transition: ...
    def persist_state(self, transition: Transition) -> Transition: ...
    def scientific_guards(self, transition: Transition) -> Transition: ...
    def verify(self, transition: Transition) -> bool: ...
    def targeted_recovery(self, transition: Transition, attempt: int) -> Transition: ...


DECODING = {"temperature": 0.2, "top_p": 0.95, "max_output_tokens": 8192,
            "max_tool_calls": 64, "max_transient_retries": 2}


def execute_ladder(backend: HarnessBackend, level: int, task: Any, state: Any,
                   model_id: str, max_local_recovery: int = 2) -> Transition:
    if level not in range(5):
        raise ValueError("H0..H4 only")
    tr = backend.core_execute(task, state, model_id, DECODING)
    if level >= 1:
        tr = backend.typed_tools(tr)
    if level >= 2:
        tr = backend.persist_state(tr)
    if level >= 3:
        tr = backend.scientific_guards(tr)
    if level >= 4:
        for attempt in range(max_local_recovery + 1):
            if backend.verify(tr):
                tr.resolved = True
                break
            if attempt < max_local_recovery:
                tr = backend.targeted_recovery(tr, attempt + 1)
    return tr
