"""Central determinism utility.

Every experiment must call seed_everything() exactly once, with the seed
value coming from that experiment's config (never hardcoded), so the seed
is recorded in the run manifest and the run is reproducible.
"""
from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed all RNGs this project currently depends on.

    Extend this function (do not create a second seeding path) if/when an
    ML framework dependency is added, so there is exactly one place that
    determinism is enforced.
    """
    random.seed(seed)
    np.random.seed(seed)
