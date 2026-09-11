"""Event-level metrics. Timestep-level metrics (AUPRC/AUROC/precision/
recall) come from sklearn directly; these functions cover the metrics that
need event structure: an onset's 1-hour pre-onset window either got at
least one alert or it didn't, and a run of consecutive positive
predictions is one alert episode, not one-per-5-minutes.
"""
from __future__ import annotations

import pandas as pd


def alert_episodes(predictions: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Collapse a boolean prediction series into (start, end) spans of
    consecutive positive predictions — one alert episode per span."""
    if predictions.empty:
        return []
    pred = predictions.astype(bool)
    is_start = pred & ~pred.shift(1, fill_value=False)
    is_end = pred & ~pred.shift(-1, fill_value=False)
    starts = pred.index[is_start]
    ends = pred.index[is_end]
    return list(zip(starts, ends))


def event_detection_rate(
    predictions: pd.Series, onset_times: list[pd.Timestamp], horizon_steps: int = 12
) -> tuple[float, list[pd.Timestamp]]:
    """Fraction of onset events with >=1 positive prediction in their
    pre-onset window (t in (onset - horizon, onset]). Returns the rate and
    the list of onsets that were missed."""
    step = predictions.index.to_series().diff().mode()[0]
    detected, missed = 0, []
    for onset in onset_times:
        if onset not in predictions.index:
            continue
        window_start = onset - step * horizon_steps
        window = predictions.loc[(predictions.index > window_start) & (predictions.index <= onset)]
        if window.astype(bool).any():
            detected += 1
        else:
            missed.append(onset)
    total = sum(1 for o in onset_times if o in predictions.index)
    rate = detected / total if total else float("nan")
    return rate, missed


def warning_lead_times(
    predictions: pd.Series, onset_times: list[pd.Timestamp], horizon_steps: int = 12
) -> list[pd.Timedelta]:
    """For each detected onset, the time between its FIRST correct alert
    within the pre-onset window and the onset itself. Undetected onsets
    are excluded (their lead time is undefined, not zero)."""
    step = predictions.index.to_series().diff().mode()[0]
    lead_times = []
    for onset in onset_times:
        if onset not in predictions.index:
            continue
        window_start = onset - step * horizon_steps
        window = predictions.loc[(predictions.index > window_start) & (predictions.index <= onset)]
        positives = window.index[window.astype(bool)]
        if len(positives) > 0:
            lead_times.append(onset - positives.min())
    return lead_times


def false_alerts_per_day(
    predictions: pd.Series, onset_times: list[pd.Timestamp], horizon_steps: int = 12
) -> float:
    """Count alert EPISODES (not raw positive timesteps) that do not
    overlap any true pre-onset window, normalised to a per-day rate."""
    step = predictions.index.to_series().diff().mode()[0]
    true_windows = []
    for onset in onset_times:
        if onset in predictions.index:
            true_windows.append((onset - step * horizon_steps, onset))

    def overlaps_any_true_window(start, end) -> bool:
        return any(start <= w_end and w_start <= end for w_start, w_end in true_windows)

    episodes = alert_episodes(predictions)
    false_episodes = [e for e in episodes if not overlaps_any_true_window(e[0], e[1])]
    n_days = (predictions.index.max() - predictions.index.min()) / pd.Timedelta(days=1)
    return len(false_episodes) / n_days if n_days > 0 else float("nan")


# --- Detection-task variants: the valid window is [onset, onset + window],
# i.e. forward from the onset, not backward before it (the pivot from
# forecasting to detection inverts which side of the onset "counts"). ---


def detection_rate_and_delay(
    predictions: pd.Series, onset_times: list[pd.Timestamp], window: pd.Timedelta = pd.Timedelta("1h")
) -> tuple[float, list[pd.Timedelta], list[pd.Timestamp]]:
    """Fraction of onsets with >=1 positive prediction in [onset, onset+window],
    the delay (alert_time - onset, >= 0) to the FIRST such alert for each
    detected onset, and the list of missed onsets."""
    detected_delays = []
    missed = []
    total = 0
    for onset in onset_times:
        if onset not in predictions.index:
            continue
        total += 1
        window_pred = predictions.loc[(predictions.index >= onset) & (predictions.index <= onset + window)]
        positives = window_pred.index[window_pred.astype(bool)]
        if len(positives) > 0:
            detected_delays.append(positives.min() - onset)
        else:
            missed.append(onset)
    rate = len(detected_delays) / total if total else float("nan")
    return rate, detected_delays, missed


def false_alerts_per_day_detection(
    predictions: pd.Series, onset_times: list[pd.Timestamp], window: pd.Timedelta = pd.Timedelta("1h")
) -> float:
    """Same as false_alerts_per_day, but for the detection task's forward
    window [onset, onset+window] rather than the forecasting task's
    backward pre-onset window."""
    true_windows = [
        (onset, onset + window) for onset in onset_times if onset in predictions.index
    ]

    def overlaps_any_true_window(start, end) -> bool:
        return any(start <= w_end and w_start <= end for w_start, w_end in true_windows)

    episodes = alert_episodes(predictions)
    false_episodes = [e for e in episodes if not overlaps_any_true_window(e[0], e[1])]
    n_days = (predictions.index.max() - predictions.index.min()) / pd.Timedelta(days=1)
    return len(false_episodes) / n_days if n_days > 0 else float("nan")
