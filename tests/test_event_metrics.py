import pandas as pd

from wds_sentinel.prediction.event_metrics import (
    alert_episodes,
    event_detection_rate,
    false_alerts_per_day,
    warning_lead_times,
)


def _index(n=60):
    return pd.date_range("2020-01-01", periods=n, freq="5min")


def test_alert_episodes_collapses_consecutive_positives():
    idx = _index(10)
    preds = pd.Series([0, 1, 1, 1, 0, 0, 1, 0, 1, 1], index=idx)
    episodes = alert_episodes(preds)
    assert len(episodes) == 3
    assert episodes[0] == (idx[1], idx[3])
    assert episodes[1] == (idx[6], idx[6])
    assert episodes[2] == (idx[8], idx[9])


def test_event_detection_rate_counts_hit_within_window():
    idx = _index(30)
    onset = idx[20]
    preds = pd.Series(0, index=idx)
    preds.loc[idx[18]] = 1  # inside the 12-step pre-onset window
    rate, missed = event_detection_rate(preds, [onset], horizon_steps=12)
    assert rate == 1.0
    assert missed == []


def test_event_detection_rate_zero_when_no_alert_in_window():
    idx = _index(30)
    onset = idx[20]
    preds = pd.Series(0, index=idx)  # never alerts
    rate, missed = event_detection_rate(preds, [onset], horizon_steps=12)
    assert rate == 0.0
    assert missed == [onset]


def test_warning_lead_time_measures_first_correct_alert():
    idx = _index(30)
    onset = idx[20]
    preds = pd.Series(0, index=idx)
    preds.loc[idx[10]] = 1  # first alert, 10 steps before onset
    preds.loc[idx[15]] = 1  # later alert, ignored for lead time
    lead_times = warning_lead_times(preds, [onset], horizon_steps=12)
    assert lead_times == [onset - idx[10]]


def test_false_alerts_per_day_ignores_true_positive_episodes():
    idx = _index(300)  # ~25 hours
    onset = idx[100]
    preds = pd.Series(0, index=idx)
    preds.loc[idx[95]] = 1  # true alert, inside pre-onset window
    preds.loc[idx[200]] = 1  # false alert, far from any onset
    rate = false_alerts_per_day(preds, [onset], horizon_steps=12)
    n_days = (idx.max() - idx.min()) / pd.Timedelta(days=1)
    assert rate == 1 / n_days


# --- detection-task metric variants ---
from wds_sentinel.prediction.event_metrics import (
    detection_rate_and_delay,
    false_alerts_per_day_detection,
)


def test_detection_rate_and_delay_measures_forward_window():
    idx = _index(30)
    onset = idx[10]
    preds = pd.Series(0, index=idx)
    preds.loc[idx[13]] = 1  # 15 min after onset -> should count, delay = 15 min
    rate, delays, missed = detection_rate_and_delay(preds, [onset], window=pd.Timedelta("1h"))
    assert rate == 1.0
    assert delays == [idx[13] - onset]
    assert missed == []


def test_detection_rate_zero_if_alert_only_before_onset():
    idx = _index(30)
    onset = idx[10]
    preds = pd.Series(0, index=idx)
    preds.loc[idx[5]] = 1  # before onset -> doesn't count for detection
    rate, delays, missed = detection_rate_and_delay(preds, [onset], window=pd.Timedelta("1h"))
    assert rate == 0.0
    assert missed == [onset]


def test_false_alerts_per_day_detection_ignores_forward_window_hits():
    idx = _index(300)
    onset = idx[100]
    preds = pd.Series(0, index=idx)
    preds.loc[idx[105]] = 1  # inside [onset, onset+1h] -> true alert
    preds.loc[idx[250]] = 1  # far away -> false alert
    rate = false_alerts_per_day_detection(preds, [onset], window=pd.Timedelta("1h"))
    n_days = (idx.max() - idx.min()) / pd.Timedelta(days=1)
    assert rate == 1 / n_days
