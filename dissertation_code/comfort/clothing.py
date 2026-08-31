"""Clothing insulation derived from outdoor conditions (DD-017).

Architecture Layer 2. PMV needs a clothing-insulation value (clo) that cannot be measured from a
T+RH-only dataset. The rest of the project inherited a *fixed* 0.5 clo from the warm-climate
LaSDPC work, which is untenable here:

    At 0.5 clo, PMV neutrality sits at ~24.5 degC. Connaught Mansions averages ~15 degC, so
    100% of readings label "too_cool" and the uncertainty trigger fires on 100% of rows. With a
    single label class there is no decision boundary, entropy is ~0 everywhere, and the
    active-learning experiment cannot run at all.

A month->clo schedule was tried and **rejected by measurement**. This building's summer indoor
mean is ~17.8 degC, not warm; lightening clothing in June drove PMV *down* (-1.40, versus -0.73
at a flat 1.0 clo). A calendar-based schedule anti-correlates the correction with the condition
it is meant to correct.

Instead, clothing responds to a running mean of **outdoor** temperature. This is standard comfort
science — ASHRAE 55 and ISO 7730 both treat clo as a contextual input rather than a constant, and
de Dear & Brager's adaptive-comfort work establishes that occupants adjust clothing in response to
outdoor conditions. It is also self-calibrating: it does not assume that "summer" means "warm".

Verified on the real export: outdoor running mean drives clo from 1.17 (January) to 0.72 (July),
yielding a stable 35-49% "too_cool" share in *every* month with no seasonal inversion.

The outdoor driver comes from the dataset's own external sensors rather than
``weather_station.csv``, which is headerless with units embedded in the values and encoding
artefacts. Using the building's own loggers keeps the provenance simple and the timestamps aligned.

**Scope note.** Outdoor temperature informs the *comfort assumption* only. It is never a model
feature — the T+RH-only constraint is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dissertation_code import config
from dissertation_code.data import schema

#: Column added by `attach_clo`, consumed by synthetic_labels.generate_labels via `clo_column`.
CLOTHING_INSULATION = "clothing_insulation"


def clo_from_outdoor(
    running_mean_outdoor: np.ndarray | pd.Series,
    clo_min: float = config.CLO_MIN,
    clo_max: float = config.CLO_MAX,
    slope: float = config.CLO_SLOPE,
    reference_temperature: float = config.CLO_REF_TEMP,
) -> np.ndarray:
    """Map an outdoor running-mean temperature (deg C) to clothing insulation (clo).

    Linear and decreasing in outdoor temperature, clipped to a plausible indoor range: occupants
    wear more when it is cold outside and less when it is mild, but never nothing and never an
    unbounded amount.

    Args:
        running_mean_outdoor: outdoor temperature running mean, deg C.
        clo_min: lightest clothing assumed.
        clo_max: heaviest clothing assumed, reached at or below `reference_temperature`.
        slope: clo lost per degC of outdoor warming.
        reference_temperature: outdoor temperature at which clo equals `clo_max`.

    Returns:
        Clothing insulation in clo, same shape as the input.
    """
    values = np.asarray(running_mean_outdoor, dtype=float)
    clo = clo_max - slope * (values - reference_temperature)
    return np.clip(clo, clo_min, clo_max)


def outdoor_running_mean(
    external: pd.DataFrame,
    days: int = config.CLO_RUNNING_MEAN_DAYS,
) -> pd.Series:
    """Daily outdoor temperature averaged across external sensors, then smoothed.

    A running mean (rather than the instantaneous reading) reflects that people dress for the
    recent weather, not for the current minute — the standard adaptive-comfort treatment.

    Args:
        external: wide frame of external-sensor readings (timestamp, zone, temperature, ...).
        days: window length of the running mean.

    Returns:
        Series indexed by `datetime.date`, holding the smoothed outdoor temperature.
    """
    if external.empty:
        raise ValueError(
            "no external-sensor readings supplied for the outdoor running mean"
        )

    daily = (
        external.assign(_date=external[schema.TIMESTAMP].dt.date)
        .groupby("_date")[schema.TEMPERATURE]
        .mean()
        .sort_index()
    )
    return daily.rolling(days, min_periods=1).mean()


def attach_clo(wide: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    """Add a per-row `clothing_insulation` column derived from outdoor conditions.

    Rows whose date has no outdoor reading (the export's gaps) fall back to the nearest available
    running-mean value rather than being dropped — clothing is a slowly varying assumption, so
    carrying the last known value forward is safer than discarding real indoor readings.

    Args:
        wide: indoor readings (schema.WIDE_COLUMNS).
        external: external-sensor readings used to derive the outdoor running mean.

    Returns:
        A copy of `wide` with the CLOTHING_INSULATION column appended.
    """
    running_mean = outdoor_running_mean(external)

    out = wide.copy()
    dates = out[schema.TIMESTAMP].dt.date
    outdoor = dates.map(running_mean)

    # Gaps: carry the last known outdoor mean forward, then back-fill the leading edge.
    if outdoor.isna().any():
        ordered = pd.Series(
            running_mean.values, index=pd.to_datetime(running_mean.index)
        )
        reindexed = (
            ordered.reindex(pd.to_datetime(pd.Index(dates.unique())).sort_values())
            .ffill()
            .bfill()
        )
        lookup = {d.date(): v for d, v in reindexed.items()}
        outdoor = dates.map(lookup)

    out[CLOTHING_INSULATION] = clo_from_outdoor(outdoor.to_numpy(dtype=float))
    return out
