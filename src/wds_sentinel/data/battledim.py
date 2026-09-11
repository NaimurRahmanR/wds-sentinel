"""Loading and onset derivation for the bundled BattLeDIM CSVs.

The CSVs are third-party research data included for reproducibility and
are not covered by the project's MIT licence. See data/PROVENANCE.md and
THIRD_PARTY_DATA_NOTICE.md.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DIR = Path("data/raw")


def load_scada_csv(path: str | Path) -> pd.DataFrame:
    """Load a BattLeDIM SCADA/Leakages CSV: ';'-separated, ','-decimal,
    a 'Timestamp' column, one column per sensor/pipe."""
    df = pd.read_csv(path, sep=";", decimal=",", parse_dates=["Timestamp"])
    return df.set_index("Timestamp").sort_index()


def derive_onsets(leak_df: pd.DataFrame) -> tuple[dict[str, pd.Timestamp], list[str]]:
    """Split a Leakages dataframe's columns into genuine onsets vs carryovers.

    A pipe is a "carryover" if it already has nonzero flow at the very
    first row of this file — i.e. the leak started before this file's time
    window began, so nothing new is being onset-detected here. Everything
    else's onset time is the first timestamp its flow becomes nonzero.

    Returns (onsets: {pipe_id: onset_timestamp}, carryover_pipe_ids).
    """
    onsets: dict[str, pd.Timestamp] = {}
    carryovers: list[str] = []
    first_row = leak_df.iloc[0]
    for col in leak_df.columns:
        if first_row[col] != 0:
            carryovers.append(col)
            continue
        nz = leak_df[col][leak_df[col] != 0]
        if len(nz) > 0:
            onsets[col] = nz.index.min()
    return onsets, carryovers
