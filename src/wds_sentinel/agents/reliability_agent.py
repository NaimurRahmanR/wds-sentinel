"""
Reliability Agent. Responsibility: decide whether the current evidence
quality is trustworthy enough to let the Reasoning Agent's decision stand
autonomously. Uses the exact same fixed cutoffs already frozen in
reliability/control.py (not re-tuned here) — this module adds per-signal
recorded reasons for explainability, it does not change the decision rule.
Equivalence with reliability.control.reliability_aware_decision is checked
directly in tests/test_reliability_agent_matches_frozen_logic.py.
"""
from __future__ import annotations

from dataclasses import dataclass

from wds_sentinel.agents.messages import EvidenceMessage, ReasoningMessage, ReliabilityMessage
from wds_sentinel.reliability.control import DISAGREEMENT_CUTOFF, MISSINGNESS_CUTOFF


@dataclass
class ReliabilityAgent:
    instability_cutoff: float  # from reliability.control.fit_instability_cutoff, computed once on clean training data
    missingness_cutoff: float = MISSINGNESS_CUTOFF
    disagreement_cutoff: float = DISAGREEMENT_CUTOFF

    def assess(self, evidence_msg: EvidenceMessage, reasoning_msg: ReasoningMessage) -> ReliabilityMessage:
        reasons = []
        if evidence_msg.missingness > self.missingness_cutoff:
            reasons.append(f"missingness {evidence_msg.missingness:.3f} > cutoff {self.missingness_cutoff}")
        if evidence_msg.disagreement > self.disagreement_cutoff:
            reasons.append(f"disagreement {evidence_msg.disagreement:.3f} > cutoff {self.disagreement_cutoff}")
        if evidence_msg.instability > self.instability_cutoff:
            reasons.append(f"instability {evidence_msg.instability:.3f} > cutoff {self.instability_cutoff:.3f}")

        unreliable = len(reasons) > 0
        if unreliable:
            final = "ESCALATE" if reasoning_msg.decision == "ALERT" else "ABSTAIN"
        else:
            final = reasoning_msg.decision

        return ReliabilityMessage(
            timestamp=reasoning_msg.timestamp, unreliable=unreliable, reasons=reasons, final_decision=final
        )
