"""Central logging configuration.

One place to configure logging so every module and every experiment run
produces logs in the same format, at a level controlled by config rather
than hardcoded per-module.
"""
from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
