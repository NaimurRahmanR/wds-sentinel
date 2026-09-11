"""Pre-specified linked extreme-weather / WDS-demand validation.

This module does not claim weather causes a hydraulic failure. It tests a narrower
and observable proposition using a real WDN dataset whose weather station is inside
the case-study network: whether extreme precipitation co-occurs with a measurable
operational demand response, and whether weather-aware prediction changes forecast
error during a frozen evaluation period.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wds_sentinel.evidence.schema import EvidenceBundle, EvidenceState
from wds_sentinel.knowledge.kbs import Fact, KnowledgeBase, ReasoningTrace, Rule
from wds_sentinel.linked_weather.bwdf import BWDFDataset

TRAIN_END = pd.Timestamp("2022-07-24 23:00", tz="Europe/Rome")
W1_START = pd.Timestamp("2022-07-25 00:00", tz="Europe/Rome")
W1_END = pd.Timestamp("2022-07-31 23:00", tz="Europe/Rome")
PRE_SPECIFIED_WEATHER_SENSITIVE_DMAS = ("DMA 2", "DMA 3")
RANDOM_SEED = 20260911


def _r8_extreme_rain(facts: dict[str, Any]) -> Fact | None:
    if facts.get("extreme_rain") is True:
        return Fact("extreme_weather_context", "EXTREME_RAIN")
    return None


def _r9_demand_suppression(facts: dict[str, Any]) -> Fact | None:
    residual = facts.get("seasonal_residual")
    cutoff = facts.get("suppression_cutoff")
    if residual is None or cutoff is None or not np.isfinite(residual) or not np.isfinite(cutoff):
        return None
    if residual <= cutoff:
        return Fact("demand_suppressed", True)
    return Fact("demand_suppressed", False)


def _r10_linked_response(facts: dict[str, Any]) -> Fact | None:
    if facts.get("extreme_rain") is True and facts.get("demand_suppressed") is True:
        return Fact("linked_weather_demand_response", True)
    return None


def _r11_linked_evidence_degraded(facts: dict[str, Any]) -> Fact | None:
    if facts.get("weather_available") is False or facts.get("demand_available") is False:
        return Fact("linked_evidence_degraded", True)
    return None


LINKED_WEATHER_RULES = [
    Rule("R8_EXTREME_RAIN", "training-calibrated extreme rainfall is present", _r8_extreme_rain),
    Rule("R9_DEMAND_SUPPRESSION", "weekly-seasonal demand residual is below its training lower-decile cutoff", _r9_demand_suppression),
    Rule("R10_LINKED_WEATHER_RESPONSE", "extreme rain co-occurs with an unusually suppressed DMA inflow", _r10_linked_response),
    Rule("R11_LINKED_EVIDENCE_DEGRADED", "weather or demand evidence is unavailable", _r11_linked_evidence_degraded),
]


@dataclass(frozen=True)
class LinkedWeatherDecision:
    timestamp: pd.Timestamp
    dma: str
    action: str
    extreme_rain: bool | None
    seasonal_residual: float | None
    suppression_cutoff: float | None
    evidence: EvidenceBundle
    reasoning: ReasoningTrace
    explanation: str


@dataclass(frozen=True)
class LinkedWeatherExperimentResult:
    rain_threshold_mm_day: float
    training_end: pd.Timestamp
    w1_start: pd.Timestamp
    w1_end: pd.Timestamp
    forecast_metrics: pd.DataFrame
    post_train_effects: pd.DataFrame
    decisions: tuple[LinkedWeatherDecision, ...]


def _daily_rain(weather: pd.DataFrame) -> pd.Series:
    # Rain is hourly depth in mm in the published BWDF data.
    return weather["Rain"].resample("1D").sum(min_count=1)


def calibrate_extreme_rain_threshold(weather_train: pd.DataFrame) -> float:
    """95th percentile of positive daily rainfall using training data only."""
    daily = _daily_rain(weather_train)
    wet = daily[daily > 0].dropna()
    if len(wet) < 20:
        raise ValueError("insufficient wet training days to calibrate an extreme-rain threshold")
    return float(wet.quantile(0.95))


def _cyclic_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour.to_numpy(dtype=float)
    dow = index.dayofweek.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        },
        index=index,
    )


def _model_features(inflow: pd.Series, weather: pd.DataFrame) -> pd.DataFrame:
    # lag168/lag336 remain outside a one-week evaluation window when forecasting W1.
    feats = pd.DataFrame(index=inflow.index)
    feats["lag_168"] = inflow.shift(168)
    feats["lag_336"] = inflow.shift(336)
    cyc = _cyclic_features(inflow.index)
    feats = feats.join(cyc)
    feats = feats.join(weather[["Rain", "Temperature", "Humidity", "Windspeed"]])
    return feats


def _forecast_one_dma(dataset: BWDFDataset, dma: str) -> dict[str, float]:
    y = dataset.inflows[dma]
    X = _model_features(y, dataset.weather)
    train_mask = y.index <= TRAIN_END
    test_mask = (y.index >= W1_START) & (y.index <= W1_END)

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    train_valid = X_train.notna().all(axis=1) & y_train.notna()
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]
    test_valid = X_test.notna().all(axis=1) & y_test.notna()

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X_train.loc[train_valid], y_train.loc[train_valid])
    pred = pd.Series(np.nan, index=X_test.index, dtype=float)
    pred.loc[test_valid] = model.predict(X_test.loc[test_valid])
    naive = y.shift(168).loc[test_mask]

    eval_valid = y_test.notna() & pred.notna() & naive.notna()
    if not eval_valid.any():
        raise ValueError(f"no valid W1 forecast rows for {dma}")
    yt = y_test.loc[eval_valid]
    yp = pred.loc[eval_valid]
    yn = naive.loc[eval_valid]
    return {
        "dma": dma,
        "n": int(eval_valid.sum()),
        "naive_mae": float(mean_absolute_error(yt, yn)),
        "weather_ridge_mae": float(mean_absolute_error(yt, yp)),
        "naive_rmse": float(mean_squared_error(yt, yn) ** 0.5),
        "weather_ridge_rmse": float(mean_squared_error(yt, yp) ** 0.5),
    }


def _training_suppression_cutoff(inflow: pd.Series) -> float:
    residual = inflow - inflow.shift(168)
    train = residual.loc[residual.index <= TRAIN_END].dropna()
    if len(train) < 100:
        raise ValueError("insufficient training residuals")
    return float(train.quantile(0.10))


def _post_train_effect(dataset: BWDFDataset, dma: str, rain_threshold: float) -> dict[str, float]:
    y = dataset.inflows[dma]
    residual = y - y.shift(168)
    daily_residual = residual.resample("1D").mean()
    daily_rain = _daily_rain(dataset.weather)
    df = pd.concat([daily_residual.rename("residual"), daily_rain.rename("rain")], axis=1)
    df = df.loc[df.index > TRAIN_END.normalize()].dropna()
    extreme = df["rain"] >= rain_threshold
    dry = df["rain"] == 0
    if extreme.sum() == 0:
        raise ValueError("no post-training extreme-rain days available")
    effect = float(df.loc[extreme, "residual"].mean() - df.loc[dry, "residual"].mean()) if dry.any() else np.nan

    # Descriptive day-level bootstrap; no causal interpretation.
    rng = np.random.default_rng(RANDOM_SEED)
    e = df.loc[extreme, "residual"].to_numpy()
    d = df.loc[dry, "residual"].to_numpy()
    if len(d) and len(e):
        boots = np.empty(2000, dtype=float)
        for i in range(2000):
            boots[i] = rng.choice(e, len(e), replace=True).mean() - rng.choice(d, len(d), replace=True).mean()
        lo, hi = np.quantile(boots, [0.025, 0.975])
    else:
        lo = hi = np.nan
    return {
        "dma": dma,
        "extreme_days": int(extreme.sum()),
        "dry_days": int(dry.sum()),
        "extreme_mean_weekly_residual_lps": float(df.loc[extreme, "residual"].mean()),
        "dry_mean_weekly_residual_lps": float(df.loc[dry, "residual"].mean()) if dry.any() else np.nan,
        "difference_lps": effect,
        "bootstrap_95_lo": float(lo),
        "bootstrap_95_hi": float(hi),
    }


def _decision_for_timestamp(
    dataset: BWDFDataset,
    dma: str,
    timestamp: pd.Timestamp,
    rain_threshold: float,
    suppression_cutoff: float,
) -> LinkedWeatherDecision:
    y = dataset.inflows[dma]
    weather = dataset.weather
    daily_rain = _daily_rain(weather)
    day = timestamp.normalize()
    rain = daily_rain.get(day, np.nan)
    y_now = y.get(timestamp, np.nan)
    y_prev = y.get(timestamp - pd.Timedelta(hours=168), np.nan)
    residual = y_now - y_prev if np.isfinite(y_now) and np.isfinite(y_prev) else np.nan
    weather_available = bool(np.isfinite(rain))
    demand_available = bool(np.isfinite(residual))
    extreme = None if not weather_available else bool(rain >= rain_threshold)

    items = (
        EvidenceState(
            source="bwdf_weather_station",
            modality="precipitation",
            timestamp=timestamp,
            value=float(rain) if weather_available else None,
            availability=weather_available,
            quality=1.0 if weather_available else 0.0,
            provenance="BWDF weather station within case-study WDN; daily rain sum",
        ),
        EvidenceState(
            source=dma,
            modality="demand_residual",
            timestamp=timestamp,
            value=float(residual) if demand_available else None,
            availability=demand_available,
            quality=1.0 if demand_available else 0.0,
            provenance="observed DMA net inflow minus same-hour previous-week inflow",
        ),
    )
    bundle = EvidenceBundle(timestamp=timestamp, items=items)
    facts = {
        "extreme_rain": extreme,
        "seasonal_residual": float(residual) if demand_available else None,
        "suppression_cutoff": suppression_cutoff,
        "weather_available": weather_available,
        "demand_available": demand_available,
    }
    trace = KnowledgeBase(LINKED_WEATHER_RULES).evaluate(facts)
    if "R11_LINKED_EVIDENCE_DEGRADED" in trace.fired_rules:
        action = "ABSTAIN"
    elif "R10_LINKED_WEATHER_RESPONSE" in trace.fired_rules:
        action = "WEATHER_LINKED_DEMAND_RESPONSE"
    elif extreme:
        action = "EXTREME_WEATHER_WATCH"
    else:
        action = "ROUTINE"
    explanation = (
        f"{dma} at {timestamp.isoformat()}: daily rainfall="
        f"{rain if weather_available else 'unavailable'} mm; extreme threshold={rain_threshold:.3f} mm; "
        f"same-hour previous-week demand residual={residual if demand_available else 'unavailable'} L/s; "
        f"suppression cutoff={suppression_cutoff:.3f} L/s; rules fired="
        f"{','.join(trace.fired_rules) if trace.fired_rules else 'none'}; action={action}. "
        "This records a co-located observational weather-demand response and does not assert causation."
    )
    return LinkedWeatherDecision(
        timestamp=timestamp,
        dma=dma,
        action=action,
        extreme_rain=extreme,
        seasonal_residual=float(residual) if demand_available else None,
        suppression_cutoff=suppression_cutoff,
        evidence=bundle,
        reasoning=trace,
        explanation=explanation,
    )


def evaluate_linked_weather_experiment(dataset: BWDFDataset) -> LinkedWeatherExperimentResult:
    weather_train = dataset.weather.loc[dataset.weather.index <= TRAIN_END]
    rain_threshold = calibrate_extreme_rain_threshold(weather_train)

    forecast_rows = [_forecast_one_dma(dataset, dma) for dma in [f"DMA {i}" for i in range(1, 11)]]
    forecast_metrics = pd.DataFrame(forecast_rows).set_index("dma")

    effect_rows = [_post_train_effect(dataset, dma, rain_threshold) for dma in PRE_SPECIFIED_WEATHER_SENSITIVE_DMAS]
    effects = pd.DataFrame(effect_rows).set_index("dma")

    # Export actual linked examples selected deterministically from post-training data:
    # first extreme-rain timestamp for each pre-specified weather-sensitive DMA.
    daily_rain = _daily_rain(dataset.weather)
    extreme_days = daily_rain[(daily_rain.index > TRAIN_END.normalize()) & (daily_rain >= rain_threshold)]
    decisions: list[LinkedWeatherDecision] = []
    if len(extreme_days):
        day = extreme_days.index[0]
        # Use 12:00 local time if available to avoid selecting a missing overnight flow by accident.
        ts = day + pd.Timedelta(hours=12)
        for dma in PRE_SPECIFIED_WEATHER_SENSITIVE_DMAS:
            decisions.append(
                _decision_for_timestamp(dataset, dma, ts, rain_threshold, _training_suppression_cutoff(dataset.inflows[dma]))
            )

    return LinkedWeatherExperimentResult(
        rain_threshold_mm_day=rain_threshold,
        training_end=TRAIN_END,
        w1_start=W1_START,
        w1_end=W1_END,
        forecast_metrics=forecast_metrics,
        post_train_effects=effects,
        decisions=tuple(decisions),
    )
