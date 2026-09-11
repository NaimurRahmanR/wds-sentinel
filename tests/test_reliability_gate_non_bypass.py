"""Structural proof, not just an outcome check: no matter what the
Reasoning Agent decides, the Supervisory Agent's final decision is
whatever the Reliability Agent says — there is no path around it."""
from dataclasses import dataclass

import pandas as pd

from wds_sentinel.agents.evidence_agent import EvidenceAgent
from wds_sentinel.agents.messages import ReliabilityMessage
from wds_sentinel.agents.prediction_agent import PredictionAgent
from wds_sentinel.agents.reasoning_agent import KnowledgeReasoningAgent
from wds_sentinel.agents.supervisory_agent import SupervisoryDecisionAgent
from wds_sentinel.evidence.adapters import DataFrameEvidenceAdapter
from wds_sentinel.knowledge.kbs import KnowledgeBase


@dataclass
class AlwaysEscalateReliabilityAgent:
    """A stub that ignores its inputs entirely and always returns
    ESCALATE — if the Supervisory Agent had any path that bypassed the
    reliability step, this test would catch it by seeing something other
    than ESCALATE come out."""
    instability_cutoff: float = 0.0

    def assess(self, evidence_msg, reasoning_msg):
        return ReliabilityMessage(
            timestamp=reasoning_msg.timestamp, unreliable=True,
            reasons=["stub: always unreliable"], final_decision="ESCALATE",
        )


def _build_supervisor(reliability_agent):
    idx = pd.date_range("2020-01-01", periods=30, freq="5min")
    pressure = pd.DataFrame({"n1": [50.0] * 30}, index=idx)
    scores = pd.Series([0.99] * 30, index=idx)  # predictor always "confident"
    eq = pd.DataFrame(
        {"missingness": [0.0] * 30, "availability": [1.0] * 30,
         "disagreement": [0.0] * 30, "instability": [0.0] * 30},
        index=idx,
    )
    corrob = pd.Series([5.0] * 30, index=idx)  # always corroborated -> reasoning would say ALERT

    evidence_agent = EvidenceAgent(
        adapters=[DataFrameEvidenceAdapter(name="p", modality="pressure", raw=pressure)],
        evidence_quality=eq,
    )
    prediction_agent = PredictionAgent(scores=scores, threshold=0.9)
    reasoning_agent = KnowledgeReasoningAgent(kb=KnowledgeBase(), corroboration=corrob)
    return SupervisoryDecisionAgent(evidence_agent, prediction_agent, reasoning_agent, reliability_agent), idx


def test_final_decision_always_equals_reliability_agents_output():
    supervisor, idx = _build_supervisor(AlwaysEscalateReliabilityAgent())
    for t in idx[:5]:
        record = supervisor.decide(t)
        # Reasoning would say ALERT (predictor confident + corroborated),
        # but the stub reliability agent always says ESCALATE — the
        # record MUST reflect ESCALATE, proving the gate cannot be skipped.
        assert record.decision == "ESCALATE"


def test_real_reliability_agent_still_produces_alert_when_actually_reliable():
    """Sanity counterpart: with the REAL reliability agent (not the stub)
    and genuinely reliable evidence, the same setup should actually reach
    ALERT — confirming the non-bypass test above isn't vacuous (i.e. it
    isn't that decisions never reach ALERT for some unrelated reason)."""
    from wds_sentinel.agents.reliability_agent import ReliabilityAgent

    supervisor, idx = _build_supervisor(ReliabilityAgent(instability_cutoff=1.0))
    record = supervisor.decide(idx[0])
    assert record.decision == "ALERT"
