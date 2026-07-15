"""Synthetic comfort-label generator (Sprint 1 / O2 deliverable).

Architecture Layer 2. Stands in for a real occupant in the proof of concept: it turns each
(temperature, RH) observation into a thermal-sensation "vote" by computing PMV and adding
Gaussian noise calibrated so the synthetic feedback *disagrees with PMV at roughly the right
rate, in roughly the right direction* (Cheung et al. 2019).

Honest limitation: this validates the system's mechanics end-to-end, NOT accuracy for a real
occupant. A real-occupant trial is future work. (See detailed_dissertation.md §1.1.2.)

Calibration rationale: Cheung et al. (2019) found PMV correctly classifies individual thermal
sensation in <34% of cases. We model occupant sensation as PMV plus zero-mean Gaussian noise;
the noise standard deviation is chosen so that, after rounding to the discrete ASHRAE 7-point
scale, the synthetic vote matches the PMV-implied category at a rate consistent with that
finding. The default sigma is exposed and documented so the calibration is auditable and tunable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dissertation_code import config
from dissertation_code.comfort import pmv as pmv_model
from dissertation_code.data import schema

# Discrete ASHRAE 7-point thermal-sensation scale.
SENSATION_SCALE = (-3, -2, -1, 0, 1, 2, 3)
SENSATION_LABELS = {
    -3: "cold",
    -2: "cool",
    -1: "slightly_cool",
    0: "neutral",
    1: "slightly_warm",
    2: "warm",
    3: "hot",
}

# Output column names (extend the unified schema, do not mutate it).
PMV_VALUE = "pmv"
SENSATION_VOTE = "sensation_vote"  # discrete ASHRAE category (-3..3)
COMFORT_LABEL = (
    "comfortable"  # bool: occupant reports comfortable (|vote| <= threshold)
)
COMFORT_CLASS = "comfort_class"  # directional class: too_cool / comfortable / too_warm


def vote_to_comfort_class(
    vote: int, threshold: int = config.COMFORT_VOTE_THRESHOLD
) -> str:
    """Map a discrete sensation vote to a directional comfort class (the model target).

    A negative vote beyond the threshold is "too cool", a positive one "too warm", otherwise
    "comfortable". The direction is what makes a recommendation actionable (warm vs cool the zone).
    """
    if vote < -threshold:
        return config.COMFORT_CLASS_TOO_COOL
    if vote > threshold:
        return config.COMFORT_CLASS_TOO_WARM
    return config.COMFORT_CLASS_COMFORTABLE


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration for synthetic label generation (defaults from the central config)."""

    noise_sigma: float = config.NOISE_STD
    seed: int = config.RANDOM_SEED
    comfort_vote_threshold: int = config.COMFORT_VOTE_THRESHOLD
    assumptions: pmv_model.ComfortAssumptions | None = None


def generate_labels(
    wide: pd.DataFrame, config: GeneratorConfig | None = None
) -> pd.DataFrame:
    """Append synthetic comfort labels to a wide (timestamp, zone, T, RH) frame.

    Deterministic for a given seed. Adds three columns: the underlying PMV, the discrete
    sensation vote (PMV + calibrated Gaussian noise, rounded to the ASHRAE scale), and a boolean
    comfort label.

    Args:
        wide: schema.WIDE_COLUMNS frame (one row per co-located T+RH observation).
        config: noise/seed/assumptions; defaults to the calibrated configuration.

    Returns:
        A copy of `wide` with PMV_VALUE, SENSATION_VOTE, COMFORT_LABEL columns added.
    """
    cfg = config or GeneratorConfig()
    required = {schema.TEMPERATURE, schema.RELATIVE_HUMIDITY}
    if not required.issubset(wide.columns):
        raise ValueError(
            f"input frame must contain {required}; got {set(wide.columns)}"
        )

    out = wide.copy()
    out[PMV_VALUE] = [
        pmv_model.pmv(t, rh, assumptions=cfg.assumptions)
        for t, rh in zip(out[schema.TEMPERATURE], out[schema.RELATIVE_HUMIDITY])
    ]

    rng = np.random.default_rng(cfg.seed)
    noisy = out[PMV_VALUE].to_numpy() + rng.normal(0.0, cfg.noise_sigma, size=len(out))
    out[SENSATION_VOTE] = _to_sensation_scale(noisy)
    out[COMFORT_LABEL] = out[SENSATION_VOTE].abs() <= cfg.comfort_vote_threshold
    out[COMFORT_CLASS] = [
        vote_to_comfort_class(v, cfg.comfort_vote_threshold)
        for v in out[SENSATION_VOTE]
    ]
    return out


def _to_sensation_scale(values: np.ndarray) -> np.ndarray:
    """Round continuous sensations to the discrete ASHRAE -3..3 scale."""
    rounded = np.rint(values).astype(int)
    return np.clip(rounded, SENSATION_SCALE[0], SENSATION_SCALE[-1])


def pmv_agreement_rate(labelled: pd.DataFrame) -> float:
    """Fraction of synthetic votes whose category equals the rounded PMV category.

    Calibration diagnostic: should sit near Cheung et al. (2019)'s ~1/3 individual-level PMV
    accuracy, confirming the noise structure is realistic rather than arbitrary.
    """
    pmv_category = _to_sensation_scale(labelled[PMV_VALUE].to_numpy())
    return float((labelled[SENSATION_VOTE].to_numpy() == pmv_category).mean())
