"""Target construction: 'will a new leak onset occur anywhere in the next
H steps', built strictly from onset timestamps that are independent of any
model output (see docs/feasibility.md Task 10 — the same non-circular
grounding principle applies to this target's label, not just to the
decision-safety definition).
"""
from __future__ import annotations

import pandas as pd


def build_onset_label(
    index: pd.DatetimeIndex, onset_times: list[pd.Timestamp], horizon_steps: int = 12
) -> pd.Series:
    """label(t) = 1 iff some onset time tau satisfies t < tau <= t + horizon_steps
    (in units of the index's own step size — 12 steps == 1 hour at 5-min
    sampling). Strictly future relative to t: an onset AT t is not itself
    "predicted in the next hour" at t.

    Implementation: mark each onset's own row, then for row t take the max
    of rows [t+1, t+horizon_steps] via a reversed rolling window (pandas
    .rolling() is backward-looking by construction, so reversing the
    series turns "next H steps" into an ordinary backward rolling window).

    NOTE: this is the forecasting-target label used by the 1-hour-ahead
    baseline, which is a documented null result (AUROC ~0.50, AUPRC ~ base
    rate — see docs/data_audit.md addendum / experiment log). Kept as-is,
    not tuned further; the live target is build_detection_label below.
    """
    indicator = pd.Series(0, index=index, dtype="int8")
    for t in onset_times:
        if t in indicator.index:
            indicator.loc[t] = 1
    shifted = indicator.shift(-1)  # bring t+1 to position t
    reversed_roll = shifted.iloc[::-1].rolling(window=horizon_steps, min_periods=1).max()
    label = reversed_roll.iloc[::-1]
    return label.fillna(0).astype("int8")


def build_detection_label(
    index: pd.DatetimeIndex, onset_times: list[pd.Timestamp], window: pd.Timedelta = pd.Timedelta("1h")
) -> pd.Series:
    """label(t) = 1 iff some onset time tau satisfies tau <= t <= tau + window
    — i.e. a leak began somewhere in L-Town within the last `window` and is
    still within its detection grace period. Unlike build_onset_label, this
    requires no knowledge of the future: by the time t >= tau, the onset has
    already happened. Detection window per onset (for evaluation) is
    exactly [tau, tau + window], matching this label definition.
    """
    label = pd.Series(0, index=index, dtype="int8")
    for tau in onset_times:
        mask = (index >= tau) & (index <= tau + window)
        label.loc[mask] = 1
    return label

