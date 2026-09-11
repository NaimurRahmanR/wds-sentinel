"""
The IDSS output artifact and its explanation. generate_explanation() is
deliberately a template over fields that already exist on the record —
it does not call a model or invent text, so the explanation is
mechanically guaranteed to only reference things that actually happened
in the trace (checked directly in test_explanation_faithfulness.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from wds_sentinel.evidence.schema import EvidenceState

VALID_DECISIONS = {"NO_ALERT", "ALERT", "ABSTAIN", "ESCALATE"}


@dataclass(frozen=True)
class DecisionRecord:
    timestamp: pd.Timestamp
    decision: str
    evidence_refs: tuple[EvidenceState, ...]
    fired_rules: tuple[str, ...]
    reliability_reasons: tuple[str, ...]
    explanation: str

    def __post_init__(self):
        if self.decision not in VALID_DECISIONS:
            raise ValueError(f"decision must be one of {VALID_DECISIONS}, got {self.decision!r}")


def generate_explanation(
    timestamp: pd.Timestamp,
    decision: str,
    fired_rules: tuple[str, ...],
    reliability_reasons: tuple[str, ...],
    evidence_refs: tuple[EvidenceState, ...],
) -> str:
    parts = [f"At {timestamp}, decision = {decision}."]
    if fired_rules:
        parts.append("Rules fired: " + ", ".join(fired_rules) + ".")
    else:
        parts.append("No rules fired.")
    if reliability_reasons:
        parts.append("Reliability override applied: " + "; ".join(reliability_reasons) + ".")
    else:
        parts.append("No reliability override applied.")
    n_avail = sum(1 for e in evidence_refs if e.availability)
    parts.append(f"Evidence: {n_avail}/{len(evidence_refs)} sources available at decision time.")
    return " ".join(parts)
