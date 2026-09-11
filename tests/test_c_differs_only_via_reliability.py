import pandas as pd

from wds_sentinel.reliability.control import is_unreliable, reliability_aware_decision


def _evidence(missingness, disagreement, instability, idx):
    return pd.DataFrame(
        {"missingness": missingness, "availability": [1 - m for m in missingness],
         "disagreement": disagreement, "instability": instability},
        index=idx,
    )


def test_c_equals_b_wherever_reliable():
    idx = pd.date_range("2020-01-01", periods=4, freq="5min")
    b = pd.Series(["ALERT", "NO_ALERT", "ALERT", "NO_ALERT"], index=idx)
    eq = _evidence(missingness=[0.0, 0.0, 0.0, 0.0], disagreement=[0.0, 0.0, 0.0, 0.0],
                    instability=[0.1, 0.1, 0.1, 0.1], idx=idx)
    c = reliability_aware_decision(b, eq, instability_cutoff=1.0)
    pd.testing.assert_series_equal(c, b)


def test_c_escalates_exactly_where_b_would_alert_and_unreliable():
    idx = pd.date_range("2020-01-01", periods=2, freq="5min")
    b = pd.Series(["ALERT", "ALERT"], index=idx)
    eq = _evidence(missingness=[0.5, 0.5], disagreement=[0.0, 0.0], instability=[0.1, 0.1], idx=idx)
    c = reliability_aware_decision(b, eq, instability_cutoff=1.0)
    assert (c == "ESCALATE").all()


def test_c_abstains_exactly_where_b_would_no_alert_and_unreliable():
    idx = pd.date_range("2020-01-01", periods=2, freq="5min")
    b = pd.Series(["NO_ALERT", "NO_ALERT"], index=idx)
    eq = _evidence(missingness=[0.5, 0.5], disagreement=[0.0, 0.0], instability=[0.1, 0.1], idx=idx)
    c = reliability_aware_decision(b, eq, instability_cutoff=1.0)
    assert (c == "ABSTAIN").all()


def test_c_never_introduces_a_decision_b_did_not_imply():
    """Exhaustive: for every combination of B in {ALERT,NO_ALERT} and
    reliability in {reliable,unreliable}, C's output is one of exactly
    the two values the mapping allows — never ALERT/NO_ALERT changed to
    the OTHER autonomous value, and never a third, undefined outcome."""
    idx = pd.date_range("2020-01-01", periods=4, freq="5min")
    b = pd.Series(["ALERT", "NO_ALERT", "ALERT", "NO_ALERT"], index=idx)
    eq = _evidence(missingness=[0.0, 0.0, 0.5, 0.5], disagreement=[0, 0, 0, 0],
                    instability=[0.1, 0.1, 0.1, 0.1], idx=idx)
    c = reliability_aware_decision(b, eq, instability_cutoff=1.0)
    expected = ["ALERT", "NO_ALERT", "ESCALATE", "ABSTAIN"]
    assert list(c) == expected


def test_is_unreliable_triggers_on_each_signal_independently():
    idx = pd.date_range("2020-01-01", periods=3, freq="5min")
    eq = _evidence(
        missingness=[0.5, 0.0, 0.0],
        disagreement=[0.0, 5.0, 0.0],
        instability=[0.1, 0.1, 10.0],
        idx=idx,
    )
    flags = is_unreliable(eq, instability_cutoff=1.0)
    assert list(flags) == [True, True, True]


def test_is_unreliable_false_when_all_signals_within_cutoffs():
    idx = pd.date_range("2020-01-01", periods=1, freq="5min")
    eq = _evidence(missingness=[0.1], disagreement=[1.0], instability=[0.5], idx=idx)
    flags = is_unreliable(eq, instability_cutoff=1.0)
    assert list(flags) == [False]
