"""
Evaluation metrics shared across systems A/B/C. "Unsafe autonomous
decision" is defined here purely from the externally-verified ground-truth
detection label (build_detection_label's output) — never from any
system's own behavior, so it means exactly the same thing when applied to
A, B, or C.
"""
from __future__ import annotations

import pandas as pd


def unsafe_autonomous_decision_rate(decisions: pd.Series, ground_truth_label: pd.Series) -> float:
    """Fraction of timesteps where the system commits to NO_ALERT
    (an autonomous claim of safety) while the externally-verified ground
    truth says a leak is within its detection window. ABSTAIN/ESCALATE
    are not autonomous claims and are excluded from the numerator by
    construction (only "NO_ALERT" counts)."""
    aligned_truth = ground_truth_label.reindex(decisions.index)
    unsafe = (decisions == "NO_ALERT") & (aligned_truth == 1)
    return float(unsafe.mean())


def coverage(decisions: pd.Series) -> float:
    """Fraction of timesteps where the system gives an autonomous
    decision (ALERT or NO_ALERT) rather than deferring (ABSTAIN/ESCALATE)."""
    return float((decisions.isin(["ALERT", "NO_ALERT"])).mean())


def abstention_escalation_rate(decisions: pd.Series) -> float:
    return float((decisions.isin(["ABSTAIN", "ESCALATE"])).mean())


def missed_event_rate(
    decisions: pd.Series, onset_times: list[pd.Timestamp], window: pd.Timedelta = pd.Timedelta("1h")
) -> tuple[float, list[pd.Timestamp]]:
    """An event counts as caught if the system outputs ALERT or ESCALATE
    at least once in [onset, onset+window] — ESCALATE still gets a human
    involved, so it counts as a catch even though it isn't an autonomous
    ALERT. Missed = neither ever occurs in the window."""
    missed = []
    total = 0
    for onset in onset_times:
        if onset not in decisions.index:
            continue
        total += 1
        window_dec = decisions.loc[(decisions.index >= onset) & (decisions.index <= onset + window)]
        if not window_dec.isin(["ALERT", "ESCALATE"]).any():
            missed.append(onset)
    rate = len(missed) / total if total else float("nan")
    return rate, missed


def alert_like_episodes(decisions: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Collapse consecutive ALERT-or-ESCALATE timesteps into episodes —
    both consume operator attention, so both count toward false-alert
    accounting."""
    is_alert_like = decisions.isin(["ALERT", "ESCALATE"])
    if not is_alert_like.any():
        return []
    is_start = is_alert_like & ~is_alert_like.shift(1, fill_value=False)
    is_end = is_alert_like & ~is_alert_like.shift(-1, fill_value=False)
    starts = decisions.index[is_start]
    ends = decisions.index[is_end]
    return list(zip(starts, ends))


def false_alerts_per_day(
    decisions: pd.Series, onset_times: list[pd.Timestamp], window: pd.Timedelta = pd.Timedelta("1h")
) -> float:
    true_windows = [(onset, onset + window) for onset in onset_times if onset in decisions.index]

    def overlaps_any(start, end) -> bool:
        return any(start <= w_end and w_start <= end for w_start, w_end in true_windows)

    episodes = alert_like_episodes(decisions)
    false_episodes = [e for e in episodes if not overlaps_any(e[0], e[1])]
    n_days = (decisions.index.max() - decisions.index.min()) / pd.Timedelta(days=1)
    return len(false_episodes) / n_days if n_days > 0 else float("nan")


def selective_unsafe_risk(
    decisions: pd.Series, ground_truth_label: pd.Series
) -> tuple[float, int, int]:
    """Selective risk = unsafe autonomous decisions / autonomous decisions.

    Denominator is exactly the autonomous-decision count (ALERT or
    NO_ALERT — ABSTAIN/ESCALATE excluded, matching coverage()'s
    definition of "autonomous"). Numerator is exactly
    unsafe_autonomous_decision_rate's numerator (NO_ALERT while ground
    truth is active), as a count rather than a rate.

    Returns (risk, n_unsafe, n_autonomous). risk is NaN — not 0, not an
    error — when n_autonomous is 0 (e.g. a policy that abstains on every
    single row): "risk" is undefined with no autonomous decisions to have
    been unsafe, and NaN is the only value that doesn't silently imply a
    coverage-free claim of safety.
    """
    aligned_truth = ground_truth_label.reindex(decisions.index)
    autonomous = decisions.isin(["ALERT", "NO_ALERT"])
    unsafe = (decisions == "NO_ALERT") & (aligned_truth == 1)
    n_autonomous = int(autonomous.sum())
    n_unsafe = int(unsafe.sum())
    risk = (n_unsafe / n_autonomous) if n_autonomous > 0 else float("nan")
    return risk, n_unsafe, n_autonomous
