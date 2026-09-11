import pandas as pd

from wds_sentinel.prediction.onset_target import build_onset_label


def _index(n=30, freq="5min"):
    return pd.date_range("2020-01-01 00:00", periods=n, freq=freq)


def test_label_is_one_for_exactly_the_hour_before_onset():
    idx = _index(30)
    onset = idx[20]  # onset at step 20
    label = build_onset_label(idx, [onset], horizon_steps=12)
    # positive for steps 8..19 inclusive (t < onset <= t+12), i.e. the 12 steps
    # strictly before the onset step
    positive_steps = set(range(8, 20))
    for i, t in enumerate(idx):
        expected = 1 if i in positive_steps else 0
        assert label.iloc[i] == expected, f"step {i} ({t}): expected {expected}, got {label.iloc[i]}"


def test_onset_itself_is_not_labeled_positive():
    idx = _index(30)
    onset = idx[20]
    label = build_onset_label(idx, [onset], horizon_steps=12)
    assert label.loc[onset] == 0


def test_no_onsets_gives_all_zero_label():
    idx = _index(30)
    label = build_onset_label(idx, [], horizon_steps=12)
    assert (label == 0).all()


def test_two_onsets_do_not_double_count_overlap():
    idx = _index(40)
    onsets = [idx[20], idx[25]]  # windows overlap (20-12=8..19, 25-12=13..24)
    label = build_onset_label(idx, onsets, horizon_steps=12)
    assert label.iloc[15] == 1  # inside both windows
    assert label.iloc[22] == 1  # inside second window only
    assert label.iloc[5] == 0  # before both windows


# --- build_detection_label ---
from wds_sentinel.prediction.onset_target import build_detection_label


def test_detection_label_positive_for_onset_and_following_hour():
    idx = _index(30)
    onset = idx[10]
    label = build_detection_label(idx, [onset], window=pd.Timedelta("55min"))  # 11 steps @5min
    positive_steps = set(range(10, 22))  # onset .. onset+55min inclusive (11 steps)
    for i, t in enumerate(idx):
        expected = 1 if i in positive_steps else 0
        assert label.iloc[i] == expected, f"step {i}: expected {expected}, got {label.iloc[i]}"


def test_detection_label_zero_before_onset():
    idx = _index(30)
    onset = idx[10]
    label = build_detection_label(idx, [onset], window=pd.Timedelta("1h"))
    assert label.iloc[:10].eq(0).all()


def test_detection_label_no_onsets_all_zero():
    idx = _index(30)
    label = build_detection_label(idx, [], window=pd.Timedelta("1h"))
    assert (label == 0).all()
