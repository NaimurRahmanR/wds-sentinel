"""Generic data-provenance utilities.

These functions know nothing about any specific dataset. Dataset-specific
facts (source URLs, licence status, expected checksums) belong in
data/PROVENANCE.md, not in code, so that a factual correction is a
documentation change rather than a code change.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_of_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(path: str | Path, expected_sha256: str) -> bool:
    """Return True iff the file's SHA-256 matches the expected value.

    Raises FileNotFoundError if the file does not exist, rather than
    silently returning False, so a missing file is never confused with a
    checksum mismatch.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return sha256_of_file(path) == expected_sha256.lower()
