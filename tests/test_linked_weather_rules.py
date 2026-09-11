import numpy as np

from wds_sentinel.knowledge.kbs import KnowledgeBase
from wds_sentinel.linked_weather.experiment import LINKED_WEATHER_RULES


def test_linked_response_requires_extreme_rain_and_suppressed_demand():
    trace = KnowledgeBase(LINKED_WEATHER_RULES).evaluate(
        {
            "extreme_rain": True,
            "seasonal_residual": -3.0,
            "suppression_cutoff": -2.0,
            "weather_available": True,
            "demand_available": True,
        }
    )
    assert "R8_EXTREME_RAIN" in trace.fired_rules
    assert "R9_DEMAND_SUPPRESSION" in trace.fired_rules
    assert "R10_LINKED_WEATHER_RESPONSE" in trace.fired_rules
    assert trace.facts["linked_weather_demand_response"] is True


def test_linked_response_does_not_assert_when_residual_not_suppressed():
    trace = KnowledgeBase(LINKED_WEATHER_RULES).evaluate(
        {
            "extreme_rain": True,
            "seasonal_residual": 1.0,
            "suppression_cutoff": -2.0,
            "weather_available": True,
            "demand_available": True,
        }
    )
    assert "R10_LINKED_WEATHER_RESPONSE" not in trace.fired_rules


def test_missing_linked_evidence_is_explicit():
    trace = KnowledgeBase(LINKED_WEATHER_RULES).evaluate(
        {
            "extreme_rain": None,
            "seasonal_residual": np.nan,
            "suppression_cutoff": -2.0,
            "weather_available": False,
            "demand_available": False,
        }
    )
    assert "R11_LINKED_EVIDENCE_DEGRADED" in trace.fired_rules
