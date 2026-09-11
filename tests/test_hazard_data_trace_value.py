from pathlib import Path

import pandas as pd

from wds_sentinel.hazard.data import TRACE_PRECIP_MM, load_daily_precipitation


def test_source_documented_trace_precipitation_is_not_missing(tmp_path: Path):
    p = tmp_path / "daily.csv"
    pd.DataFrame(
        {"Precp": [-9.8, 0.0, 12.5]},
        index=pd.date_range("2018-01-01", periods=3, freq="D"),
    ).to_csv(p)

    series = load_daily_precipitation([p])
    assert series.iloc[0] == TRACE_PRECIP_MM == 0.09
    assert series.notna().all()
