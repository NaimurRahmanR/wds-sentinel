from wds_sentinel.knowledge.kbs import KnowledgeBase
from wds_sentinel.reasoning.rules import CORROBORATION_COLUMNS  # noqa: F401 (documents the reuse link)


def test_deterministic_same_facts_same_trace():
    kb = KnowledgeBase()
    facts = {"predictor_score": 0.99, "predictor_threshold": 0.9, "corroboration_count": 2.0}
    t1 = kb.evaluate(facts)
    t2 = kb.evaluate(facts)
    assert t1.fired_rules == t2.fired_rules
    assert t1.facts == t2.facts
    assert t1.contradictions == t2.contradictions


def test_alert_requires_both_predictor_and_corroboration():
    kb = KnowledgeBase()
    # predictor alert, corroborated -> ALERT
    t = kb.evaluate({"predictor_score": 0.99, "predictor_threshold": 0.9, "corroboration_count": 1.0})
    assert t.facts["hybrid_decision"] == "ALERT"
    assert "contradiction:predictor_alert_without_corroboration" not in t.contradictions

    # predictor alert, NOT corroborated -> NO_ALERT + recorded contradiction
    t2 = kb.evaluate({"predictor_score": 0.99, "predictor_threshold": 0.9, "corroboration_count": 0.0})
    assert t2.facts["hybrid_decision"] == "NO_ALERT"
    assert "contradiction:predictor_alert_without_corroboration" in t2.contradictions

    # predictor below threshold -> NO_ALERT, no contradiction regardless of corroboration
    t3 = kb.evaluate({"predictor_score": 0.1, "predictor_threshold": 0.9, "corroboration_count": 5.0})
    assert t3.facts["hybrid_decision"] == "NO_ALERT"
    assert t3.contradictions == []


def test_missing_predictor_score_produces_no_decision_not_a_crash():
    kb = KnowledgeBase()
    t = kb.evaluate({"predictor_score": None, "predictor_threshold": 0.9, "corroboration_count": 1.0})
    assert "predictor_alert" not in t.facts
    assert "hybrid_decision" not in t.facts
    # R2_CORROBORATED is independent of predictor_score and correctly still
    # fires; R1/R3/R4 all depend on predictor_alert and correctly don't.
    assert t.fired_rules == ["R2_CORROBORATED"]


def test_rules_are_identifiable_and_fixed_order():
    kb = KnowledgeBase()
    t = kb.evaluate({"predictor_score": 0.99, "predictor_threshold": 0.9, "corroboration_count": 0.0})
    assert t.fired_rules == [
        "R1_PREDICTOR_ALERT", "R2_CORROBORATED", "R3_HYBRID_DECISION", "R4_CONTRADICTION_CHECK",
    ]


def test_kbs_hybrid_decision_matches_frozen_reasoning_rules_logic():
    """The KBS must not silently diverge from the already-validated
    reasoning/rules.py hybrid_decision logic it's meant to formalise."""
    import pandas as pd

    from wds_sentinel.reasoning.rules import hybrid_decision

    scores = pd.Series([0.99, 0.99, 0.1])
    features = pd.DataFrame({
        "cross_count_below_z_cutoff": [1, 0, 0],
        "flow_cross_count_below_z_cutoff": [0, 0, 0],
        "level_cross_count_below_z_cutoff": [0, 0, 0],
    })
    frozen = hybrid_decision(scores, features, threshold=0.9)

    kb = KnowledgeBase()
    for i in range(3):
        corrob = (
            features["cross_count_below_z_cutoff"].iloc[i]
            + features["flow_cross_count_below_z_cutoff"].iloc[i]
            + features["level_cross_count_below_z_cutoff"].iloc[i]
        )
        trace = kb.evaluate(
            {"predictor_score": scores.iloc[i], "predictor_threshold": 0.9, "corroboration_count": corrob}
        )
        assert trace.facts["hybrid_decision"] == frozen.iloc[i]
