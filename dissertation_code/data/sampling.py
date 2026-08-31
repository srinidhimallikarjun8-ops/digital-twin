"""Active-learning pool construction (architecture Layer 1).

Two jobs, both about making the experiment tractable and honest:

**Sub-sampling.** The Bath export is ~420k indoor readings at a 5-minute cadence, so adjacent
rows are near-duplicates carrying almost no additional information. A stratified 20k pool keeps
``predict_proba`` at roughly half a second, which is what makes a multi-seed, multi-strategy
experiment matrix feasible; the full export would buy nothing but runtime. Sampling is stratified
by (zone x month) so the pool preserves the real room mix and seasonal spread rather than
over-representing whichever room happens to have the longest record.

**Trigger precomputation.** ``sustained_humidity_trigger`` costs ~4.3 s per call at 50k rows and
``select_queries`` calls it on *every* active-learning iteration — roughly 7 minutes of pure
recomputation per run, for values that never change. Computing the trigger columns once here and
having the query selector read them turns that into a one-off cost.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from dissertation_code import config
from dissertation_code.comfort import clothing
from dissertation_code.comfort import pmv as pmv_model
from dissertation_code.comfort import synthetic_labels as sl
from dissertation_code.data import schema

logger = logging.getLogger(__name__)

#: Boolean column: PMV lies outside the ASHRAE comfort band for this row.
PMV_TRIGGERED = "pmv_triggered"
#: Boolean column: relative humidity has been sustained above threshold for long enough.
RH_TRIGGERED = "rh_triggered"
#: Boolean column: either domain trigger fired.
ANY_TRIGGERED = "any_triggered"
#: Calendar month, retained for stratification and for per-season analysis.
MONTH = "month"


def attach_month(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the calendar-month column used for stratification and seasonal analysis."""
    out = frame.copy()
    out[MONTH] = out[schema.TIMESTAMP].dt.month
    return out


def attach_triggers(labelled: pd.DataFrame) -> pd.DataFrame:
    """Precompute the domain-trigger columns once for the whole pool.

    PMV is reused from the labelling step when present (it is computed there with the row's own
    clothing insulation), so the trigger and the labels agree by construction. Recomputing it
    here with default assumptions would silently evaluate the trigger at a different clo.
    """
    from dissertation_code.model import active_learning as al

    out = labelled.copy()

    if sl.PMV_VALUE in out.columns:
        pmv_values = out[sl.PMV_VALUE].to_numpy()
    elif clothing.CLOTHING_INSULATION in out.columns:
        pmv_values = pmv_model.pmv_series(
            out[schema.TEMPERATURE],
            out[schema.RELATIVE_HUMIDITY],
            out[clothing.CLOTHING_INSULATION],
        )
    else:  # pragma: no cover - fixed-assumption fallback for the LaSDPC path
        pmv_values = np.array(
            [
                pmv_model.pmv(t, rh)
                for t, rh in zip(out[schema.TEMPERATURE], out[schema.RELATIVE_HUMIDITY])
            ]
        )

    out[PMV_TRIGGERED] = (
        np.abs(pmv_values - config.PMV_NEUTRAL) > config.PMV_NEUTRAL_BAND
    )
    out[RH_TRIGGERED] = al.sustained_humidity_trigger(out).to_numpy()
    out[ANY_TRIGGERED] = out[PMV_TRIGGERED] | out[RH_TRIGGERED]

    logger.info(
        "trigger base rates: pmv=%.1f%% rh=%.1f%% any=%.1f%%",
        100 * out[PMV_TRIGGERED].mean(),
        100 * out[RH_TRIGGERED].mean(),
        100 * out[ANY_TRIGGERED].mean(),
    )
    return out


def build_pool(
    labelled: pd.DataFrame,
    n: int = config.POOL_SIZE,
    random_state: int = config.RANDOM_SEED,
    stratify_by: tuple[str, ...] = (schema.ZONE, MONTH),
) -> pd.DataFrame:
    """Draw a stratified active-learning pool with trigger columns precomputed.

    Args:
        labelled: the full labelled wide frame.
        n: target pool size. Returns everything if the input is smaller.
        random_state: seed for the draw.
        stratify_by: columns defining the strata; each is sampled proportionally.

    Returns:
        A pool frame, chronologically sorted, carrying the label columns plus `month` and the
        three trigger columns.
    """
    frame = attach_month(labelled)

    if len(frame) <= n:
        pool = frame.copy()
    else:
        fraction = n / len(frame)
        rng = np.random.default_rng(random_state)
        # Sample stratum by stratum: `groupby.apply` is deprecated for this and would drop the
        # grouping columns from the result on current pandas.
        parts = [
            group.sample(
                # At least one row per stratum so no (room, month) cell vanishes.
                n=max(1, round(len(group) * fraction)),
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            for _, group in frame.groupby(list(stratify_by), sort=True)
        ]
        pool = pd.concat(parts, ignore_index=True)

    pool = pool.sort_values([schema.ZONE, schema.TIMESTAMP]).reset_index(drop=True)
    pool = attach_triggers(pool)

    logger.info(
        "built pool: %d rows, %d zones, %d months, class balance %s",
        len(pool),
        pool[schema.ZONE].nunique(),
        pool[MONTH].nunique(),
        pool[sl.COMFORT_CLASS].value_counts(normalize=True).round(3).to_dict(),
    )
    return pool
