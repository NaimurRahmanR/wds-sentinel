import pandas as pd

from wds_sentinel.agents.decision_record import generate_explanation
from wds_sentinel.knowledge.kbs import DEFAULT_RULES
from wds_sentinel.evidence.schema import EvidenceState

ALL_RULE_IDS = [r.rule_id for r in DEFAULT_RULES]


def test_explanation_only_mentions_rules_that_actually_fired():
    t = pd.Timestamp("2020-01-01")
    fired = ("R1_PREDICTOR_ALERT", "R2_CORROBORATED")
    not_fired = [rid for rid in ALL_RULE_IDS if rid not in fired]
    explanation = generate_explanation(t, "ALERT", fired, (), ())

    for rid in fired:
        assert rid in explanation
    for rid in not_fired:
        assert rid not in explanation


def test_explanation_only_mentions_reasons_actually_given():
    t = pd.Timestamp("2020-01-01")
    reasons = ("missingness 0.500 > cutoff 0.2",)
    explanation = generate_explanation(t, "ABSTAIN", (), reasons, ())
    assert "missingness 0.500 > cutoff 0.2" in explanation
    assert "disagreement" not in explanation  # not given as a reason here, must not appear
    assert "instability" not in explanation


def test_explanation_evidence_count_matches_actual_bundle():
    t = pd.Timestamp("2020-01-01")
    evidence = (
        EvidenceState("p1", "pressure", t, 50.0, True, 1.0),
        EvidenceState("p2", "pressure", t, None, False, 0.0),
        EvidenceState("p3", "pressure", t, 51.0, True, 1.0),
    )
    explanation = generate_explanation(t, "NO_ALERT", (), (), evidence)
    assert "2/3 sources available" in explanation


def test_explanation_says_no_override_when_none_applied():
    t = pd.Timestamp("2020-01-01")
    explanation = generate_explanation(t, "ALERT", ("R1_PREDICTOR_ALERT",), (), ())
    assert "No reliability override applied." in explanation


def test_explanation_decision_field_matches_the_record_decision():
    t = pd.Timestamp("2020-01-01")
    for decision in ["NO_ALERT", "ALERT", "ABSTAIN", "ESCALATE"]:
        explanation = generate_explanation(t, decision, (), (), ())
        assert f"decision = {decision}" in explanation
