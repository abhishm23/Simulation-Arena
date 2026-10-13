"""Deterministic randomness.

Rule: the same (seed, use_case_id, model_key) ALWAYS produces the same numbers.
No global `np.random.*`, no time-based seeds, no dict-ordering effects.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _offset(seed: int, *tags: str) -> int:
    """Stable 64-bit offset derived from the seed and string tags."""
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode("utf-8"))
    for t in tags:
        h.update(b"|")
        h.update(str(t).encode("utf-8"))
    return int.from_bytes(h.digest(), "big")


def make_rng(seed: int, *tags: str) -> np.random.Generator:
    """Independent, reproducible stream per (seed, tags)."""
    return np.random.Generator(np.random.PCG64(_offset(seed, *tags)))


def seed_fingerprint(seed: int, *tags: str) -> str:
    """Short human-visible fingerprint so users can verify reproducibility."""
    return format(_offset(seed, *tags) & 0xFFFFFF, "06X")
