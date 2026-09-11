from pathlib import Path

import numpy as np
import pandas as pd

from wds_sentinel.linked_weather.bwdf import DMA_NAMES, WEATHER_NAMES, load_bwdf_dataset


def _write_fixture(path: Path) -> None:
    idx = pd.date_range("2022-07-01", periods=72, freq="h")
    inflow = pd.DataFrame({"Datetime": idx})
    for i in range(10):
        inflow[f"raw_{i}"] = np.arange(len(idx), dtype=float) + i
    weather = pd.DataFrame(
        {
            "Datetime": idx,
            "rain": np.zeros(len(idx)),
            "temp": np.full(len(idx), 25.0),
            "humidity": np.full(len(idx), 60.0),
            "wind": np.full(len(idx), 5.0),
        }
    )
    with pd.ExcelWriter(path / "InflowData.xlsx", engine="openpyxl", datetime_format="DD/MM/YYYY HH:MM") as writer:
        inflow.to_excel(writer, index=False)
    with pd.ExcelWriter(path / "WeatherData.xlsx", engine="openpyxl", datetime_format="DD/MM/YYYY HH:MM") as writer:
        weather.to_excel(writer, index=False)


def test_bwdf_loader_preserves_aligned_real_evidence_contract(tmp_path: Path):
    _write_fixture(tmp_path)
    ds = load_bwdf_dataset(tmp_path)
    assert list(ds.inflows.columns) == DMA_NAMES
    assert list(ds.weather.columns) == WEATHER_NAMES
    assert ds.inflows.index.equals(ds.weather.index)
    assert str(ds.index.tz) == "Europe/Rome"
    assert len(ds.index) == 72


def test_bwdf_loader_refuses_missing_pair(tmp_path: Path):
    pd.DataFrame({"Datetime": [pd.Timestamp("2022-07-01")], "x": [1]}).to_excel(
        tmp_path / "InflowData.xlsx", index=False
    )
    try:
        load_bwdf_dataset(tmp_path)
    except FileNotFoundError as exc:
        assert "WeatherData.xlsx" in str(exc)
    else:
        raise AssertionError("expected missing paired weather file to fail")
