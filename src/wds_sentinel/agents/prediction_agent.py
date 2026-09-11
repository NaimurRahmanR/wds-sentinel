"""
Prediction Agent. Responsibility: report the frozen predictor's score for
a timestamp. It does not retrain, retune, or otherwise touch the frozen
predictive experiment — it reads from a scores Series computed once by
wds_sentinel.prediction.frozen_predictor / reliability.systems.run_systems
and already validated in the reliability experiment.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from wds_sentinel.agents.messages import PredictionMessage


@dataclass
class PredictionAgent:
    scores: pd.Series          # precomputed by the frozen predictor, not recomputed here
    threshold: float             # the frozen predictor's own threshold, unchanged
    predictor_version: str = "frozen_predictor_v1"

    def predict(self, timestamp: pd.Timestamp) -> PredictionMessage:
        score = float(self.scores.loc[timestamp]) if timestamp in self.scores.index else float("nan")
        return PredictionMessage(
            timestamp=timestamp, score=score, threshold=self.threshold,
            predictor_version=self.predictor_version,
        )
