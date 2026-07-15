"""Reproducibility helper (backend guidelines §7).

Call `make_reproducible()` once at every entry point (main, notebooks, evaluation runs) so that
the same input and seed always produce the same output.
"""

from __future__ import annotations

import os
import random

import numpy as np

from dissertation_code import config


def make_reproducible(seed: int = config.RANDOM_SEED) -> int:
    """Seed Python, numpy, and the hash randomisation for deterministic runs.

    sklearn estimators are seeded separately by passing ``random_state=seed`` at construction;
    this helper covers the global sources of nondeterminism.

    Args:
        seed: the integer seed; defaults to ``config.RANDOM_SEED``.

    Returns:
        The seed that was applied (for logging into a run manifest).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed
