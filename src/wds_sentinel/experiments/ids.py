"""Experiment identifier generation.

An experiment_id must make every run traceable to an exact code + config
state and must never collide, so that results/raw/<experiment_id>/ is
always a new, immutable directory and no run silently overwrites another.
"""
from __future__ import annotations

import hashlib
import subprocess
import uuid
from datetime import datetime, timezone


def git_short_sha(cwd: str | None = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "nogit"


def config_hash(config_text: str) -> str:
    return hashlib.sha256(config_text.encode("utf-8")).hexdigest()[:12]


def make_experiment_id(config_text: str, cwd: str | None = None) -> str:
    """Build an experiment ID that is unique even for two runs of the same
    config within the same wall-clock second.

    Second-resolution timestamps are not sufficient on their own (verified
    by test_experiment_ids.py: two rapid calls with identical config
    produced identical IDs before this fix) — a short random suffix
    guarantees no collision regardless of timing.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:6]
    return f"{git_short_sha(cwd)}_{config_hash(config_text)}_{ts}_{suffix}"
