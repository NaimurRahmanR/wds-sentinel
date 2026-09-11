"""
Reliability-aware control for system C. Computes four evidence-quality
signals — missingness, availability, disagreement, instability — none of
which ever reference the ground-truth label (checked directly in
tests/test_no_ground_truth_leakage.py, not just asserted by convention).

Fixed a priori, never tuned against A/B/C's comparative results:
  - missingness cutoff: 0.20 (more than 20% of the 37 raw sensors missing)
  - disagreement cutoff: 3.0 (group mean z-scores disagree by more than
    3 standard-deviation-equivalent units — an extension of the same -2
    per-sensor z convention already used elsewhere, not a new invented
    number)
  - instability cutoff: the 99th percentile of instability observed
    during CLEAN 2018 TRAINING only, computed once and frozen — "worse
    than the model has ever reliably seen," not an arbitrary constant
"""
from __future__ import annotations

import pandas as pd

MISSINGNESS_CUTOFF = 0.20
DISAGREEMENT_CUTOFF = 3.0
INSTABILITY_PERCENTILE = 99.0

GROUP_Z_COLUMNS = {"pressure": "cross_mean_z", "flow": "flow_cross_mean_z", "level": "level_cross_mean_z"}


def compute_evidence_quality(
    raw_pressure: pd.DataFrame, raw_flow: pd.DataFrame, raw_level: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    """raw_* are the (possibly degraded) RAW sensor readings, BEFORE any
    imputation — missingness must be measured on what's actually missing,
    not on an imputed stand-in. `features` is the (post-imputation)
    engineered frame the frozen predictor consumes, reused here only for
    its already-computed group z-score and diff1 columns."""
    all_raw = pd.concat([raw_pressure, raw_flow, raw_level], axis=1)
    missingness = all_raw.isna().mean(axis=1)
    availability = 1.0 - missingness

    group_z = pd.concat(
        [features[col].rename(name) for name, col in GROUP_Z_COLUMNS.items()], axis=1
    )
    pairs = [("pressure", "flow"), ("pressure", "level"), ("flow", "level")]
    disagreement = pd.concat(
        [(group_z[a] - group_z[b]).abs() for a, b in pairs], axis=1
    ).max(axis=1)

    diff1_cols = [c for c in features.columns if c.endswith("_diff1")]
    instability = features[diff1_cols].std(axis=1, skipna=True)

    return pd.DataFrame(
        {
            "missingness": missingness,
            "availability": availability,
            "disagreement": disagreement,
            "instability": instability,
        }
    )


def fit_instability_cutoff(clean_training_features: pd.DataFrame) -> float:
    """Computed ONCE on clean 2018 training data and frozen; never
    recomputed per-condition or re-tuned against results."""
    diff1_cols = [c for c in clean_training_features.columns if c.endswith("_diff1")]
    instability = clean_training_features[diff1_cols].std(axis=1, skipna=True)
    return float(instability.quantile(INSTABILITY_PERCENTILE / 100.0))


def is_unreliable(evidence_quality: pd.DataFrame, instability_cutoff: float) -> pd.Series:
    return (
        (evidence_quality["missingness"] > MISSINGNESS_CUTOFF)
        | (evidence_quality["disagreement"] > DISAGREEMENT_CUTOFF)
        | (evidence_quality["instability"] > instability_cutoff)
    )


def reliability_aware_decision(
    base_decision: pd.Series, evidence_quality: pd.DataFrame, instability_cutoff: float
) -> pd.Series:
    """System C: identical to B's base_decision wherever evidence is
    reliable; overridden to ESCALATE (if B would ALERT) or ABSTAIN (if B
    would NO_ALERT) wherever it isn't. This is the ONLY difference between
    B and C — verified directly in tests/test_c_differs_only_via_reliability.py."""
    unreliable = is_unreliable(evidence_quality, instability_cutoff)
    decisions = []
    for t in base_decision.index:
        if unreliable.loc[t]:
            decisions.append("ESCALATE" if base_decision.loc[t] == "ALERT" else "ABSTAIN")
        else:
            decisions.append(base_decision.loc[t])
    return pd.Series(decisions, index=base_decision.index)
