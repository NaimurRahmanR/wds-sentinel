import pandas as pd

from wds_sentinel.agents.evidence_agent import EvidenceAgent
from wds_sentinel.agents.hazard_agent import HazardAgent
from wds_sentinel.agents.integrated_supervisory_agent import IntegratedSupervisoryAgent
from wds_sentinel.agents.prediction_agent import PredictionAgent
from wds_sentinel.agents.reasoning_agent import KnowledgeReasoningAgent
from wds_sentinel.agents.reliability_agent import ReliabilityAgent
from wds_sentinel.agents.supervisory_agent import SupervisoryDecisionAgent
from wds_sentinel.evidence.adapters import DataFrameEvidenceAdapter, HazardEvidenceAdapter
from wds_sentinel.knowledge.kbs import DEFAULT_RULES, KnowledgeBase


def _wds_supervisor(n=20):
    idx = pd.date_range("2020-01-01", periods=n, freq="5min")
    pressure = pd.DataFrame({"n1": [50.0] * n}, index=idx)
    scores = pd.Series([0.99 if i % 4 == 0 else 0.1 for i in range(n)], index=idx)
    eq = pd.DataFrame(
        {"missingness": [0.0] * n, "availability": [1.0] * n,
         "disagreement": [0.0] * n, "instability": [0.5] * n}, index=idx,
    )
    corrob = pd.Series([1.0 if i % 4 == 0 else 0.0 for i in range(n)], index=idx)
    evidence_agent = EvidenceAgent(
        adapters=[DataFrameEvidenceAdapter("pressure", "pressure", pressure)], evidence_quality=eq,
    )
    prediction_agent = PredictionAgent(scores=scores, threshold=0.9)
    reasoning_agent = KnowledgeReasoningAgent(kb=KnowledgeBase(), corroboration=corrob)
    reliability_agent = ReliabilityAgent(instability_cutoff=1.0)
    return SupervisoryDecisionAgent(evidence_agent, prediction_agent, reasoning_agent, reliability_agent), idx


def _hazard_agent(idx):
    flags = pd.Series([True if i % 7 == 0 else False for i in range(len(idx))], index=idx, dtype=object)
    raw = pd.Series([10.0] * len(idx), index=idx)
    return HazardAgent(adapter=HazardEvidenceAdapter(name="hazard", elevated_flag=flags, raw_value=raw))


def test_wds_decision_inside_integrated_agent_is_bit_identical_to_standalone():
    """The exact 'unchanged behaviour of existing frozen WDS-only paths'
    proof: run the same SupervisoryDecisionAgent both standalone and
    wrapped inside IntegratedSupervisoryAgent, and confirm identical
    output for every timestamp."""
    supervisor, idx = _wds_supervisor()
    hazard_agent = _hazard_agent(idx)
    integrated = IntegratedSupervisoryAgent(wds_supervisor=supervisor, hazard_agent=hazard_agent)

    for t in idx:
        standalone_record = supervisor.decide(t)
        integrated_record = integrated.decide(t)
        assert integrated_record.wds_record == standalone_record
        assert integrated_record.decision == standalone_record.decision


def test_default_rules_unchanged_four_rules_same_ids():
    """Guards against accidental future modification of the frozen WDS
    rule set while adding hazard rules elsewhere."""
    assert len(DEFAULT_RULES) == 4
    assert [r.rule_id for r in DEFAULT_RULES] == [
        "R1_PREDICTOR_ALERT", "R2_CORROBORATED", "R3_HYBRID_DECISION", "R4_CONTRADICTION_CHECK",
    ]


def test_integrated_record_produces_all_five_required_scenario_fields():
    supervisor, idx = _wds_supervisor()
    hazard_agent = _hazard_agent(idx)
    integrated = IntegratedSupervisoryAgent(wds_supervisor=supervisor, hazard_agent=hazard_agent)
    record = integrated.decide(idx[0])
    assert record.wds_record is not None
    assert record.hazard_message is not None
    assert record.hazard_trace is not None
    assert isinstance(record.combined_explanation, str) and len(record.combined_explanation) > 0
    assert record.decision in {"NO_ALERT", "ALERT", "ABSTAIN", "ESCALATE"}


def test_combined_explanation_faithfulness_no_hallucinated_hazard_rules():
    """Same faithfulness standard as the WDS explanation: the combined
    explanation only mentions hazard rules that actually fired."""
    supervisor, idx = _wds_supervisor()
    hazard_agent = _hazard_agent(idx)
    integrated = IntegratedSupervisoryAgent(wds_supervisor=supervisor, hazard_agent=hazard_agent)

    elevated_t = idx[0]  # i%7==0 -> elevated True
    normal_t = idx[1]     # elevated False
    rec_elevated = integrated.decide(elevated_t)
    rec_normal = integrated.decide(normal_t)

    assert "R5_HAZARD_ELEVATED" in rec_elevated.combined_explanation
    assert "R6_HAZARD_NORMAL" not in rec_elevated.combined_explanation

    assert "R6_HAZARD_NORMAL" in rec_normal.combined_explanation
    assert "R5_HAZARD_ELEVATED" not in rec_normal.combined_explanation


def test_combined_explanation_never_claims_hazard_changes_wds_decision():
    supervisor, idx = _wds_supervisor()
    hazard_agent = _hazard_agent(idx)
    integrated = IntegratedSupervisoryAgent(wds_supervisor=supervisor, hazard_agent=hazard_agent)
    record = integrated.decide(idx[0])
    assert "does not alter the WDS operational state" in record.combined_explanation


def test_integrated_agent_supports_independent_wds_and_hazard_clocks():
    supervisor, wds_idx = _wds_supervisor()
    hazard_idx = pd.date_range("2018-01-01", periods=len(wds_idx), freq="D")
    hazard_agent = _hazard_agent(hazard_idx)
    integrated = IntegratedSupervisoryAgent(wds_supervisor=supervisor, hazard_agent=hazard_agent)

    record = integrated.decide(wds_idx[0], hazard_idx[1])
    assert record.wds_timestamp == wds_idx[0]
    assert record.hazard_timestamp == hazard_idx[1]
    assert record.hazard_message.timestamp == hazard_idx[1]
    assert record.wds_execution.timestamp == wds_idx[0]
