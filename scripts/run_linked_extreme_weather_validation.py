#!/usr/bin/env python
"""Run the pre-specified physically linked BWDF weather-demand validation."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from wds_sentinel.linked_weather import load_bwdf_dataset, evaluate_linked_weather_experiment

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "external" / "bwdf"
DERIVED = ROOT / "results" / "derived"
FIGURES = ROOT / "results" / "figures"
RAW = ROOT / "results" / "raw"


def _serialise_decision(d):
    return {
        "timestamp": d.timestamp.isoformat(),
        "dma": d.dma,
        "action": d.action,
        "extreme_rain": d.extreme_rain,
        "seasonal_residual": d.seasonal_residual,
        "suppression_cutoff": d.suppression_cutoff,
        "rules_fired": d.reasoning.fired_rules,
        "facts": d.reasoning.facts,
        "evidence": [
            {
                "source": e.source,
                "modality": e.modality,
                "timestamp": e.timestamp.isoformat(),
                "value": e.value,
                "availability": e.availability,
                "quality": e.quality,
                "provenance": e.provenance,
            }
            for e in d.evidence.items
        ],
        "explanation": d.explanation,
    }


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    dataset = load_bwdf_dataset(DATA_DIR)
    result = evaluate_linked_weather_experiment(dataset)

    result.forecast_metrics.to_csv(DERIVED / "linked_bwdf_w1_forecast_metrics.csv")
    result.post_train_effects.to_csv(DERIVED / "linked_bwdf_extreme_rain_effects.csv")
    payload = {
        "rain_threshold_mm_day": result.rain_threshold_mm_day,
        "training_end": result.training_end.isoformat(),
        "w1_start": result.w1_start.isoformat(),
        "w1_end": result.w1_end.isoformat(),
        "decisions": [_serialise_decision(d) for d in result.decisions],
    }
    (DERIVED / "linked_bwdf_decision_traces.json").write_text(json.dumps(payload, indent=2, default=str))

    # Forecast comparison figure (all 10 DMAs; highlight the pre-specified linked pair in the CSV/report).
    fm = result.forecast_metrics
    ax = fm[["naive_mae", "weather_ridge_mae"]].plot(kind="bar", figsize=(11, 5))
    ax.set_ylabel("W1 MAE (L/s)")
    ax.set_title("BWDF W1: seasonal-naive vs weather-aware Ridge")
    ax.legend(["Previous-week baseline", "Weather-aware Ridge"])
    plt.tight_layout()
    plt.savefig(FIGURES / "linked_bwdf_w1_forecast_comparison.png", dpi=180)
    plt.close()

    # Physically linked effect figure for the pre-specified weather-sensitive DMAs.
    ef = result.post_train_effects
    ax = ef[["extreme_mean_weekly_residual_lps", "dry_mean_weekly_residual_lps"]].plot(
        kind="bar", figsize=(7, 5)
    )
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Mean demand minus previous-week same-hour demand (L/s)")
    ax.set_title("Co-located extreme-rain vs dry-day demand response")
    ax.legend(["Extreme-rain days", "Dry days"])
    plt.tight_layout()
    plt.savefig(FIGURES / "linked_bwdf_extreme_rain_response.png", dpi=180)
    plt.close()

    summary_lines = [
        "Linked BWDF extreme-weather validation",
        f"Training end: {result.training_end}",
        f"W1: {result.w1_start} to {result.w1_end}",
        f"Training-calibrated extreme-rain threshold: {result.rain_threshold_mm_day:.6f} mm/day",
        "",
        "W1 forecast metrics:",
        result.forecast_metrics.to_string(),
        "",
        "Post-training linked effects (pre-specified DMA 2/3):",
        result.post_train_effects.to_string(),
        "",
        "Decision traces:",
    ]
    for d in result.decisions:
        summary_lines.append(d.explanation)
    text = "\n".join(summary_lines) + "\n"
    (RAW / "linked_bwdf_extreme_weather_validation.log").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
