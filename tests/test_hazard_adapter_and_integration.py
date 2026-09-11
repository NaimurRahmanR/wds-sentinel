import pandas as pd

from wds_sentinel.agents.hazard_agent import HazardAgent
from wds_sentinel.evidence.adapters import HazardEvidenceAdapter
from wds_sentinel.knowledge.kbs import DEFAULT_RULES, HAZARD_RULES, KnowledgeBase


def _adapter():
    idx = pd.date_range("2018-01-01", periods=5, freq="D")
    flags = pd.Series([True, False, pd.NA, False, True], index=idx, dtype=object)
    raw = pd.Series([50.0, 0.0, float("nan"), 1.0, 80.0], index=idx)
    return HazardEvidenceAdapter(name="taiwan_precip", elevated_flag=flags, raw_value=raw)


def test_hazard_adapter_schema_matches_evidence_state_contract():
    from wds_sentinel.evidence.schema import EvidenceState

    adapter = _adapter()
    idx = adapter.elevated_flag.index
    items = adapter.fetch(idx[0])
    assert len(items) == 1
    assert isinstance(items[0], EvidenceState)
    assert items[0].modality == "hazard"
    assert items[0].availability is True
    assert items[0].value == 1.0


def test_hazard_adapter_reports_unavailable_when_flag_undefined():
    adapter = _adapter()
    idx = adapter.elevated_flag.index
    item = adapter.fetch(idx[2])[0]
    assert item.availability is False
    assert item.value is None
    assert item.quality == 0.0


def test_hazard_agent_reports_elevated_normal_and_unavailable_correctly():
    agent = HazardAgent(adapter=_adapter())
    idx = agent.adapter.elevated_flag.index
    assert agent.collect(idx[0]).elevated is True
    assert agent.collect(idx[1]).elevated is False
    msg_undefined = agent.collect(idx[2])
    assert msg_undefined.available is False
    assert msg_undefined.elevated is None


def test_hazard_rules_fire_independently_of_wds_default_rules():
    """HAZARD_RULES must never appear when evaluating with only
    DEFAULT_RULES, and vice versa — proving the two rule sets are
    genuinely separate, not silently merged."""
    wds_only_kb = KnowledgeBase()  # exactly the existing, unmodified default
    trace = wds_only_kb.evaluate({"hazard_elevated": True, "hazard_available": True})
    assert trace.fired_rules == []  # DEFAULT_RULES know nothing about hazard facts

    hazard_only_kb = KnowledgeBase(HAZARD_RULES)
    trace2 = hazard_only_kb.evaluate({"hazard_elevated": True, "hazard_available": True})
    assert trace2.fired_rules == ["R5_HAZARD_ELEVATED"]
    assert "R1_PREDICTOR_ALERT" not in trace2.fired_rules  # WDS rules absent from this KB


def test_hazard_evidence_degraded_rule_fires_only_when_unavailable():
    kb = KnowledgeBase(HAZARD_RULES)
    t_available = kb.evaluate({"hazard_elevated": False, "hazard_available": True})
    assert "R7_HAZARD_EVIDENCE_DEGRADED" not in t_available.fired_rules

    t_unavailable = kb.evaluate({"hazard_elevated": None, "hazard_available": False})
    assert "R7_HAZARD_EVIDENCE_DEGRADED" in t_unavailable.fired_rules
