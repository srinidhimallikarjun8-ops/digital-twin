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
    vote: int,
    threshold: int = config.COMFORT_VOTE_THRESHOLD,
    merge_warm: bool = False,
) -> str:
    """Map a discrete sensation vote to a directional comfort class (the model target).

    A negative vote beyond the threshold is "too cool", a positive one "too warm", otherwise
    "comfortable". The direction is what makes a recommendation actionable (warm vs cool the zone).

    Args:
        vote: discrete ASHRAE sensation vote (-3..+3).
        threshold: |vote| <= this is "comfortable".
        merge_warm: collapse "too_warm" into "comfortable" (DD-019). Required for the Bath
            building, which never exceeds 22.9 degC: at that hottest reading in 8 months PMV is
            -0.08, so P(vote > +1) = 0.058 even there. Every "too_warm" label would be a
            sigma=1.0 noise draw with no physical signal, and class_weight="balanced" would
            up-weight that pure noise ~150x. Left False so existing 3-class behaviour is unchanged.
    """
    if vote < -threshold:
        return config.COMFORT_CLASS_TOO_COOL
    if vote > threshold and not merge_warm:
        return config.COMFORT_CLASS_TOO_WARM
    return config.COMFORT_CLASS_COMFORTABLE


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration for synthetic label generation (defaults from the central config)."""

    noise_sigma: float = config.NOISE_STD
    seed: int = config.RANDOM_SEED
    comfort_vote_threshold: int = config.COMFORT_VOTE_THRESHOLD
    assumptions: pmv_model.ComfortAssumptions | None = None
    #: Column holding a per-row clothing insulation (clo). When set, PMV is computed row-wise
    #: with that value instead of the single fixed `assumptions` clo — see comfort/clothing.py
    #: and DD-017. None preserves the original fixed-assumption behaviour exactly.
    clo_column: str | None = None
    #: Collapse "too_warm" into "comfortable" (DD-019). See `vote_to_comfort_class`.
    merge_warm: bool = False
    #: Fraction of the noise variance that is a *persistent per-occupant offset* rather than
    #: independent per-reading jitter (DD-023). None reproduces the original all-independent
    #: behaviour exactly.
    #:
    #: The default model draws fresh noise for every reading, which implies the same occupant
    #: feels randomly different every five minutes. Real occupants are internally consistent:
    #: an individual who runs cold runs cold all day. Comfort research treats that as
    #: between-person variance, distinct from within-person variability.
    #:
    #: Splitting the variance keeps the *marginal* distribution — and therefore the Cheung et
    #: al. (2019) agreement calibration — unchanged, because
    #: ``sigma_between^2 + sigma_within^2 = noise_sigma^2``.
    between_occupant_fraction: float | None = None
    #: Column identifying the occupant. Each distinct value receives its own persistent offset.
    occupant_column: str = schema.ZONE


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
    if cfg.clo_column is not None:
        if cfg.clo_column not in out.columns:
            raise ValueError(
                f"clo_column {cfg.clo_column!r} not found in the input frame; "
                "call comfort.clothing.attach_clo first"
            )
        out[PMV_VALUE] = pmv_model.pmv_series(
            out[schema.TEMPERATURE],
            out[schema.RELATIVE_HUMIDITY],
            out[cfg.clo_column],
        )
    else:
        out[PMV_VALUE] = [
            pmv_model.pmv(t, rh, assumptions=cfg.assumptions)
            for t, rh in zip(out[schema.TEMPERATURE], out[schema.RELATIVE_HUMIDITY])
        ]

    rng = np.random.default_rng(cfg.seed)
    noise = _draw_noise(out, cfg, rng)
    noisy = out[PMV_VALUE].to_numpy() + noise
    out[SENSATION_VOTE] = _to_sensation_scale(noisy)
    out[COMFORT_LABEL] = out[SENSATION_VOTE].abs() <= cfg.comfort_vote_threshold
    out[COMFORT_CLASS] = [
        vote_to_comfort_class(v, cfg.comfort_vote_threshold, merge_warm=cfg.merge_warm)
        for v in out[SENSATION_VOTE]
    ]
    return out


def _draw_noise(
    frame: pd.DataFrame, cfg: GeneratorConfig, rng: np.random.Generator
) -> np.ndarray:
    """Draw the occupant-response noise added to PMV.

    Two structures, both with total standard deviation ``cfg.noise_sigma``:

    * **Independent** (default) — a fresh draw per reading. Simple, but implies an occupant whose
      thermal sensation is uncorrelated between one reading and the next.
    * **Split** (``between_occupant_fraction`` set) — a persistent offset per occupant plus
      independent jitter, with ``sigma_between^2 + sigma_within^2 = noise_sigma^2``.

    The split preserves the marginal noise distribution exactly, so the PMV-agreement rate that
    calibrates ``noise_sigma`` against Cheung et al. (2019) is unaffected. What changes is that
    part of the deviation is now *structured* — constant within an occupant — and therefore
    learnable from that occupant's readings, rather than irreducible.
    """
    if cfg.between_occupant_fraction is None:
        return rng.normal(0.0, cfg.noise_sigma, size=len(frame))

    fraction = cfg.between_occupant_fraction
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"between_occupant_fraction must be in [0, 1]; got {fraction}")
    if cfg.occupant_column not in frame.columns:
        raise ValueError(
            f"occupant_column {cfg.occupant_column!r} not found in the input frame"
        )

    # Variance split, so the marginal standard deviation stays at noise_sigma.
    sigma_between = cfg.noise_sigma * np.sqrt(fraction)
    sigma_within = cfg.noise_sigma * np.sqrt(1.0 - fraction)

    occupants = frame[cfg.occupant_column].to_numpy()
    unique = pd.unique(occupants)
    offsets = dict(zip(unique, rng.normal(0.0, sigma_between, size=len(unique))))

    persistent = np.array([offsets[o] for o in occupants])
    jitter = rng.normal(0.0, sigma_within, size=len(frame))
    return persistent + jitter


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
