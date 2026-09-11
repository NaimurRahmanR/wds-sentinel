"""
Explicit, deterministic evidence-combination rule for the HYBRID system
(B). Combines the frozen predictor's score with an independent,
already-computed statistical corroboration signal (how many sensors are
themselves currently anomalous by a fixed, standard z-score convention) —
not a new hydraulic threshold, a reuse of the same -2 z convention already
established in prediction/features.py's build_change_features.

B's decision space is still only {NO_ALERT, ALERT} — the reliability
control that adds ABSTAIN/ESCALATE is a separate layer (reliability/control.py)
applied on top of this same rule for system C, not part of this module.
"""
from __future__ import annotations

import pandas as pd

CORROBORATION_COLUMNS = [
    "cross_count_below_z_cutoff",
    "flow_cross_count_below_z_cutoff",
    "level_cross_count_below_z_cutoff",
]


def corroboration_count(features: pd.DataFrame) -> pd.Series:
    """Total number of sensors (summed across all three streams) whose
    z-score is currently below the standard -2 cutoff — an independent,
    already-computed anomaly signal, not derived from the predictor."""
    return features[CORROBORATION_COLUMNS].sum(axis=1)


def direct_decision(scores: pd.Series, threshold: float) -> pd.Series:
    """System A: the predictor's threshold alone."""
    return pd.Series(
        ["ALERT" if s >= threshold else "NO_ALERT" for s in scores], index=scores.index
    )


def hybrid_decision(scores: pd.Series, features: pd.DataFrame, threshold: float) -> pd.Series:
    """System B: ALERT only if the predictor says ALERT AND at least one
    sensor is independently, statistically corroborating that signal.
    Reduces false alarms relative to A at the cost of possibly missing
    weakly-corroborated true events — a stated trade-off, not tuned to
    any particular outcome."""
    corrob = corroboration_count(features)
    decisions = []
    for t in scores.index:
        predictor_alert = scores.loc[t] >= threshold
        corroborated = corrob.loc[t] >= 1
        decisions.append("ALERT" if (predictor_alert and corroborated) else "NO_ALERT")
    return pd.Series(decisions, index=scores.index)
