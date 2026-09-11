import pandas as pd
import pytest

from wds_sentinel.agents.decision_record import DecisionRecord
from wds_sentinel.agents.evidence_agent import EvidenceAgent
from wds_sentinel.agents.messages import EvidenceMessage, PredictionMessage, ReasoningMessage
from wds_sentinel.agents.prediction_agent import PredictionAgent
from wds_sentinel.agents.reasoning_agent import KnowledgeReasoningAgent
from wds_sentinel.agents.reliability_agent import ReliabilityAgent
from wds_sentinel.agents.supervisory_agent import SupervisoryDecisionAgent
from wds_sentinel.evidence.adapters import DataFrameEvidenceAdapter
from wds_sentinel.knowledge.kbs import KnowledgeBase


def _full_pipeline(n=20):
    idx = pd.date_range("2020-01-01", periods=n, freq="5min")
    pressure = pd.DataFrame({"n1": [50.0] * n, "n2": [40.0] * n}, index=idx)
    scores = pd.Series([0.99 if i % 5 == 0 else 0.1 for i in range(n)], index=idx)
    eq = pd.DataFrame(
        {"missingness": [0.0] * n, "availability": [1.0] * n,
         "disagreement": [0.0] * n, "instability": [0.5] * n},
        index=idx,
    )
    corrob = pd.Series([1.0 if i % 5 == 0 else 0.0 for i in range(n)], index=idx)

    evidence_agent = EvidenceAgent(
        adapters=[DataFrameEvidenceAdapter(name="pressure", modality="pressure", raw=pressure)],
        evidence_quality=eq,
    )
    prediction_agent = PredictionAgent(scores=scores, threshold=0.9)
    reasoning_agent = KnowledgeReasoningAgent(kb=KnowledgeBase(), corroboration=corrob)
    reliability_agent = ReliabilityAgent(instability_cutoff=1.0)
    supervisor = SupervisoryDecisionAgent(evidence_agent, prediction_agent, reasoning_agent, reliability_agent)
    return supervisor, idx


def test_each_agent_returns_its_declared_message_type():
    supervisor, idx = _full_pipeline()
    t = idx[0]
    ev_msg = supervisor.evidence_agent.collect(t)
    assert isinstance(ev_msg, EvidenceMessage)
    pred_msg = supervisor.prediction_agent.predict(t)
    assert isinstance(pred_msg, PredictionMessage)
    reasoning_msg = supervisor.reasoning_agent.reason(pred_msg, ev_msg)
    assert isinstance(reasoning_msg, ReasoningMessage)
    rel_msg = supervisor.reliability_agent.assess(ev_msg, reasoning_msg)
    assert rel_msg.final_decision in {"NO_ALERT", "ALERT", "ABSTAIN", "ESCALATE"}


def test_supervisor_produces_a_valid_decision_record_for_every_timestep():
    supervisor, idx = _full_pipeline()
    for t in idx:
        record = supervisor.decide(t)
        assert isinstance(record, DecisionRecord)
        assert record.decision in {"NO_ALERT", "ALERT", "ABSTAIN", "ESCALATE"}
        assert record.timestamp == t
        assert isinstance(record.explanation, str) and len(record.explanation) > 0


def test_decision_record_rejects_invalid_decision_value():
    t = pd.Timestamp("2020-01-01")
    with pytest.raises(ValueError):
        DecisionRecord(t, "MAYBE", (), (), (), "bad decision")


def test_end_to_end_alert_then_no_alert_pattern_matches_expected_schedule():
    """Confirms the whole wiring produces the expected pattern given the
    synthetic setup's known schedule (predictor+corroboration both fire
    every 5th step)."""
    supervisor, idx = _full_pipeline()
    decisions = [supervisor.decide(t).decision for t in idx]
    for i, d in enumerate(decisions):
        if i % 5 == 0:
            assert d == "ALERT", f"step {i}: expected ALERT, got {d}"
        else:
            assert d == "NO_ALERT", f"step {i}: expected NO_ALERT, got {d}"
