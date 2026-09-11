import pandas as pd

from wds_sentinel.data.battledim import derive_onsets


def test_genuine_onset_detected_when_zero_at_file_start():
    idx = pd.date_range("2019-01-01", periods=10, freq="5min")
    df = pd.DataFrame({"pA": [0, 0, 0, 5, 5, 5, 5, 5, 5, 5]}, index=idx)
    onsets, carryovers = derive_onsets(df)
    assert carryovers == []
    assert onsets["pA"] == idx[3]


def test_carryover_detected_when_nonzero_at_file_start():
    idx = pd.date_range("2019-01-01", periods=10, freq="5min")
    df = pd.DataFrame({"pB": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]}, index=idx)
    onsets, carryovers = derive_onsets(df)
    assert carryovers == ["pB"]
    assert onsets == {}


def test_pipe_never_active_gives_neither_onset_nor_carryover():
    idx = pd.date_range("2019-01-01", periods=10, freq="5min")
    df = pd.DataFrame({"pC": [0] * 10}, index=idx)
    onsets, carryovers = derive_onsets(df)
    assert carryovers == []
    assert onsets == {}
