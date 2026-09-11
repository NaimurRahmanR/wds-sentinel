"""
Knowledge/Reasoning Agent. Responsibility: turn (predictor score,
corroboration evidence) into a base ALERT/NO_ALERT decision via the
explicit, identifiable KnowledgeBase rules — never ABSTAIN/ESCALATE,
which only the Reliability Agent can introduce.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from wds_sentinel.agents.messages import EvidenceMessage, PredictionMessage, ReasoningMessage
from wds_sentinel.knowledge.kbs import KnowledgeBase


@dataclass
class KnowledgeReasoningAgent:
    kb: KnowledgeBase
    corroboration: pd.Series  # precomputed via reasoning.rules.corroboration_count, not recomputed here

    def reason(self, prediction_msg: PredictionMessage, evidence_msg: EvidenceMessage) -> ReasoningMessage:
        t = prediction_msg.timestamp
        corrob = float(self.corroboration.loc[t]) if t in self.corroboration.index else 0.0
        facts = {
            "predictor_score": prediction_msg.score if prediction_msg.score == prediction_msg.score else None,  # NaN check
            "predictor_threshold": prediction_msg.threshold,
            "corroboration_count": corrob,
        }
        trace = self.kb.evaluate(facts)
        decision = trace.facts.get("hybrid_decision", "NO_ALERT")
        return ReasoningMessage(timestamp=t, trace=trace, decision=decision)
