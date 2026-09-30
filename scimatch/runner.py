"""Online orchestration. Neither native evaluator nor final SVR enters control state."""
from dataclasses import dataclass
from typing import Any, Callable
from .controller import SciMatchController
from .features import Capability, Prefix
from .harness import execute_ladder, Transition
from .adapters import NativeBenchmark, HeldOutSVR


@dataclass
class RunResult:
    transitions: list[Transition]
    decisions: list
    final_evaluation: Any
    stopped_reason: str


def run_task(task, state, model_id: str, capability: Capability,
             controller: SciMatchController, benchmark: NativeBenchmark,
             svr_evaluator: HeldOutSVR,
             prefix_from_state: Callable[[Any], Prefix],
             update_state: Callable[[Any, Transition], Any],
             remaining_cost: Callable[[Any], float],
             escalate_model: Callable[[str], tuple[str, Capability]] | None = None) -> RunResult:
    """One task. The SVR evaluator is called only after termination.

    A replayable official benchmark backend supplies guards, recovery and
    task-specific completion; missing native hooks raise rather than simulating
    benchmark results. Model escalation is limited to one tier per task.
    """
    transitions, decisions, escalations = [], [], 0
    backend = benchmark.backend()
    risk_adapter = benchmark.public_risk_adapter()
    reason = "horizon"
    evidence = []
    while controller.ledger.committed < controller.config.max_decisions:
        budget = remaining_cost(state)
        risk = risk_adapter.risk(evidence, budget)
        dec = controller.decide(capability, risk, prefix_from_state(state), budget)
        decisions.append(dec)
        if dec.level is None:
            # Recovery is attempted by H4; only a feasible re-decision may commit.
            for _ in range(controller.config.max_local_recovery):
                tr = execute_ladder(backend, 4, task, state, model_id, 1)
                evidence = tr.evidence
                risk = risk_adapter.risk(evidence, remaining_cost(state))
                dec = controller.decide(capability, risk, prefix_from_state(state),
                                        remaining_cost(state))
                decisions.append(dec)
                if dec.level is not None:
                    break
            if dec.level is None and escalate_model and escalations < controller.config.max_model_escalations:
                model_id, capability = escalate_model(model_id)
                escalations += 1
                dec = controller.decide(capability, risk, prefix_from_state(state),
                                        remaining_cost(state))
                decisions.append(dec)
            if dec.level is None:
                reason = "inadmissible_after_recovery"
                break
        tr = execute_ladder(backend, dec.level, task, state, model_id,
                            controller.config.max_local_recovery)
        # The introduced-failure ground truth is a post-commit training label;
        # the live controller does not read it or use it to decide whether to commit.
        controller.commit(dec, transition_committed=True)
        transitions.append(tr)
        state = update_state(state, tr)
        evidence = tr.evidence
        if getattr(state, "complete", False):
            reason = "complete"
            break
    # Evaluation-only verifier receives final artifact after controller stops.
    final_artifact = transitions[-1].artifact if transitions else None
    evaluation = svr_evaluator.evaluate(task, final_artifact, transitions)
    return RunResult(transitions, decisions, evaluation, reason)
