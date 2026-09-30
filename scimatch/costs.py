"""Keep provider inference, tokens, tool usage, wall time, and search calls distinct."""
from dataclasses import dataclass


@dataclass
class Usage:
    acting_provider_usd: float = 0.0
    proposer_provider_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    wall_seconds: float = 0.0
    development_feedback_outcomes: int = 0

    def add(self, other: "Usage"):
        for key in self.__dataclass_fields__:
            setattr(self, key, getattr(self, key) + getattr(other, key))

    def reprice(self, multiplier: float):
        if multiplier <= 0:
            raise ValueError("positive price multiplier required")
        return Usage(self.acting_provider_usd*multiplier,
                     self.proposer_provider_usd*multiplier,
                     self.input_tokens, self.output_tokens, self.tool_calls,
                     self.wall_seconds, self.development_feedback_outcomes)


def relative_cost(usage: Usage, h0_reference: float) -> float:
    if h0_reference <= 0:
        raise ValueError("H0 provider cost must be positive")
    return usage.acting_provider_usd / h0_reference


def matched_evaluation_cost(a: Usage, b: Usage, tolerance: float):
    """Match expected evaluation-time acting provider cost; audit other fields."""
    return abs(a.acting_provider_usd - b.acting_provider_usd) <= tolerance
