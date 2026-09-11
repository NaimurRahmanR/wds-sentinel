import pandas as pd

from wds_sentinel.reliability.evaluation import coverage, selective_unsafe_risk


def _idx(n=10):
    return pd.date_range("2020-01-01", periods=n, freq="5min")


def test_denominator_excludes_abstain_and_escalate():
    idx = _idx(6)
    decisions = pd.Series(["ALERT", "NO_ALERT", "ABSTAIN", "ESCALATE", "NO_ALERT", "ALERT"], index=idx)
    truth = pd.Series(0, index=idx)
    risk, n_unsafe, n_autonomous = selective_unsafe_risk(decisions, truth)
    assert n_autonomous == 4  # ALERT, NO_ALERT, NO_ALERT, ALERT — not the 2 deferred ones
    assert n_autonomous == int(coverage(decisions) * len(idx))


def test_numerator_matches_unsafe_autonomous_decision_rate_definition():
    idx = _idx(4)
    decisions = pd.Series(["NO_ALERT", "NO_ALERT", "ALERT", "ABSTAIN"], index=idx)
    truth = pd.Series([1, 0, 1, 1], index=idx)  # leak active at steps 0,2,3
    # unsafe = NO_ALERT while truth==1 -> only step 0 qualifies (step1: NO_ALERT but truth=0;
    # step2: ALERT so not unsafe; step3: ABSTAIN, not autonomous, excluded from both counts)
    risk, n_unsafe, n_autonomous = selective_unsafe_risk(decisions, truth)
    assert n_unsafe == 1
    assert n_autonomous == 3  # NO_ALERT, NO_ALERT, ALERT (ABSTAIN excluded)
    assert risk == 1 / 3


def test_zero_coverage_gives_nan_not_zero_or_error():
    idx = _idx(5)
    decisions = pd.Series(["ABSTAIN"] * 3 + ["ESCALATE"] * 2, index=idx)
    truth = pd.Series([1, 0, 1, 0, 1], index=idx)
    risk, n_unsafe, n_autonomous = selective_unsafe_risk(decisions, truth)
    assert n_autonomous == 0
    assert n_unsafe == 0
    assert risk != risk  # NaN != NaN is the standard way to check for NaN
    assert not (risk == 0.0)  # must not silently collapse to a "safe-looking" zero


def test_full_coverage_all_safe_gives_zero_risk_not_nan():
    idx = _idx(3)
    decisions = pd.Series(["ALERT", "ALERT", "NO_ALERT"], index=idx)
    truth = pd.Series([1, 1, 0], index=idx)  # no NO_ALERT-during-active-leak anywhere
    risk, n_unsafe, n_autonomous = selective_unsafe_risk(decisions, truth)
    assert n_autonomous == 3
    assert n_unsafe == 0
    assert risk == 0.0


def test_ground_truth_reindexed_safely_if_index_differs():
    idx_dec = _idx(3)
    decisions = pd.Series(["NO_ALERT", "NO_ALERT", "NO_ALERT"], index=idx_dec)
    truth_full = pd.Series([1, 1, 1, 1, 1], index=_idx(5))  # longer index, decisions is a subset
    risk, n_unsafe, n_autonomous = selective_unsafe_risk(decisions, truth_full)
    assert n_autonomous == 3
    assert n_unsafe == 3
    assert risk == 1.0
