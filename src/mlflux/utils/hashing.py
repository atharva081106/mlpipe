"""Hashing utilities for data integrity and artifact verification."""

import hashlib
from pathlib import Path
from typing import Union


def compute_file_hash(path: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file."""
    path = Path(path)
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
