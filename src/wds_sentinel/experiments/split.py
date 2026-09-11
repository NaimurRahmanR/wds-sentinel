"""Temporal train/validation/test split, with an embargo (purge) window at
each boundary so that no labeled example's forward-looking horizon reaches
across a split boundary into the next partition.

Boundaries (frozen for this experiment — see docs/research_protocol_status.md):
  train:      2018-01-01 00:00 -> 2018-06-30 23:55   (moved from the
              originally-proposed 2018-09-30 boundary: the onset-based
              target had only 2 validation events there vs. 5 here — see
              docs/feasibility.md addendum for the event-count comparison)
  validation: 2018-07-01 00:00 -> 2018-12-31 23:55
  test:       2019-01-01 00:00 -> 2019-12-31 23:55   (final temporal period; earlier rejected forecasting-baseline exposure is disclosed)
"""
from __future__ import annotations

import pandas as pd

TRAIN_START = pd.Timestamp("2018-01-01 00:00:00")
TRAIN_END = pd.Timestamp("2018-06-30 23:55:00")
VAL_START = pd.Timestamp("2018-07-01 00:00:00")
VAL_END = pd.Timestamp("2018-12-31 23:55:00")
TEST_START = pd.Timestamp("2019-01-01 00:00:00")
TEST_END = pd.Timestamp("2019-12-31 23:55:00")


def split_masks(
    index: pd.DatetimeIndex, horizon_steps: int = 12
) -> dict[str, pd.Series]:
    """Boolean masks for train/validation/test, each with the last
    horizon_steps rows before its upper boundary excluded (embargoed) so
    that no example's onset-label horizon can reach into the next
    partition's territory.

    This embargo direction (end-of-partition) is correct for a
    FORWARD-looking label (build_onset_label). For the backward-looking
    detection label (build_detection_label), use split_masks_detection
    instead — the leakage direction is mirrored.
    """
    step = index.to_series().diff().mode()[0]
    embargo = step * horizon_steps

    train = (index >= TRAIN_START) & (index <= TRAIN_END - embargo)
    val = (index >= VAL_START) & (index <= VAL_END - embargo)
    test = (index >= TEST_START) & (index <= TEST_END)  # nothing follows test; no embargo needed

    return {
        "train": pd.Series(train, index=index),
        "validation": pd.Series(val, index=index),
        "test": pd.Series(test, index=index),
    }


def split_masks_detection(
    index: pd.DatetimeIndex, window: pd.Timedelta = pd.Timedelta("1h")
) -> dict[str, pd.Series]:
    """Boolean masks for train/validation/test for the BACKWARD-looking
    detection label: a positive label at t depends on an onset up to
    `window` in the past, so the leakage risk is at the START of
    validation/test (a label there could stem from an onset whose distinct
    pressure signature the model already saw as a training/validation
    positive example) — the mirror image of split_masks' embargo.
    """
    train = (index >= TRAIN_START) & (index <= TRAIN_END)
    val = (index >= VAL_START + window) & (index <= VAL_END)
    test = (index >= TEST_START + window) & (index <= TEST_END)

    return {
        "train": pd.Series(train, index=index),
        "validation": pd.Series(val, index=index),
        "test": pd.Series(test, index=index),
    }
