"""Loader for the Battle of Water Demand Forecasting (BWDF) supplementary data.

The BWDF data describe a real WDN in north-east Italy. Hourly net inflow for
10 DMAs and weather observations from a station located within the case-study
WDN share the same time axis. The data are distributed under CC BY 4.0 by the
BWDF authors / Water-Futures implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DMA_NAMES = [f"DMA {i}" for i in range(1, 11)]
WEATHER_NAMES = ["Rain", "Temperature", "Humidity", "Windspeed"]


@dataclass(frozen=True)
class BWDFDataset:
    inflows: pd.DataFrame
    weather: pd.DataFrame

    def __post_init__(self) -> None:
        if not self.inflows.index.equals(self.weather.index):
            raise ValueError("BWDF inflow and weather indexes must align exactly")
        if list(self.inflows.columns) != DMA_NAMES:
            raise ValueError(f"unexpected inflow columns: {list(self.inflows.columns)!r}")
        if list(self.weather.columns) != WEATHER_NAMES:
            raise ValueError(f"unexpected weather columns: {list(self.weather.columns)!r}")
        if not isinstance(self.inflows.index, pd.DatetimeIndex):
            raise TypeError("BWDF indexes must be DatetimeIndex")

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.inflows.index


def _read_bwdf_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(
        path,
        parse_dates=[0],
        date_format="%d/%m/%Y %H:%M",
        index_col=0,
        na_values=["", " ", "NULL", "null", "-", "NaN", "nan"],
    )
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("Europe/Rome", ambiguous="infer", nonexistent="shift_forward")
    else:
        idx = idx.tz_convert("Europe/Rome")
    df.index = idx
    df.index.name = "Datetime"
    return df


def load_bwdf_dataset(data_dir: str | Path) -> BWDFDataset:
    """Load bundled BWDF ``InflowData.xlsx`` and ``WeatherData.xlsx``.

    No interpolation or imputation is performed here. Missing observations remain
    missing so downstream reliability logic can see evidence availability.
    """
    data_dir = Path(data_dir)
    inflow_path = data_dir / "InflowData.xlsx"
    weather_path = data_dir / "WeatherData.xlsx"
    if not inflow_path.exists() or not weather_path.exists():
        missing = [str(p.name) for p in (inflow_path, weather_path) if not p.exists()]
        raise FileNotFoundError(f"missing BWDF file(s): {', '.join(missing)}")

    inflows = _read_bwdf_excel(inflow_path)
    weather = _read_bwdf_excel(weather_path)
    if inflows.shape[1] != 10:
        raise ValueError(f"expected 10 DMA inflow columns, got {inflows.shape[1]}")
    if weather.shape[1] != 4:
        raise ValueError(f"expected 4 weather columns, got {weather.shape[1]}")
    inflows.columns = DMA_NAMES
    weather.columns = WEATHER_NAMES

    # The official packaged files are intended to share one hourly time axis.
    # Refuse silent alignment/truncation because that would weaken the physical link.
    return BWDFDataset(inflows=inflows, weather=weather)
