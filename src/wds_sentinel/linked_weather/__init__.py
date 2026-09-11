"""Physically linked weather-demand validation using the BWDF real WDN dataset."""

from .bwdf import BWDFDataset, load_bwdf_dataset
from .experiment import (
    LINKED_WEATHER_RULES,
    LinkedWeatherDecision,
    LinkedWeatherExperimentResult,
    calibrate_extreme_rain_threshold,
    evaluate_linked_weather_experiment,
)

__all__ = [
    "BWDFDataset",
    "load_bwdf_dataset",
    "LINKED_WEATHER_RULES",
    "LinkedWeatherDecision",
    "LinkedWeatherExperimentResult",
    "calibrate_extreme_rain_threshold",
    "evaluate_linked_weather_experiment",
]
