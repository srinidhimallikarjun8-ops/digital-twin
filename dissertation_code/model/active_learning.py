"""Active-learning loop, uncertainty scoring, and query triggers (architecture Layer 3).

The learner queries the (synthetic) occupant only at high-uncertainty moments — the algorithmic
embodiment of the heritage minimum-intervention principle (detailed_dissertation.md §2.4). An
instance is a query candidate when EITHER the model is uncertain (entropy/margin) OR a domain
trigger fires: predicted comfort outside +/-0.5 PMV from neutral, OR relative humidity sustained
above 75% for over 30 minutes.

Fallback (risk register): if active learning does not help, ``train_static_baseline`` trains a
plain Random Forest on all available labels — the interface and evaluation contributions remain
intact without a rewrite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dissertation_code import config
from dissertation_code.comfort import pmv as pmv_model
from dissertation_code.comfort import synthetic_labels as sl
from dissertation_code.data import schema
from dissertation_code.model.base import ComfortModel

logger = logging.getLogger(__name__)


# --- Uncertainty scoring -------------------------------------------------------------------
def uncertainty_score(
    proba: np.ndarray, strategy: str = config.UNCERTAINTY_STRATEGY
) -> np.ndarray:
    """Per-sample uncertainty from a class-probability matrix.

    Args:
        proba: shape (n_samples, n_classes) probabilities.
        strategy: "entropy" (Shannon entropy) or "margin" (1 - gap between top two classes).

    Returns:
        Uncertainty in [0, 1]-ish range; higher means more uncertain.
    """
    if strategy == "entropy":
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.where(proba > 0, np.log(proba), 0.0)
        entropy = -np.sum(proba * logs, axis=1)
        return entropy / np.log(proba.shape[1])  # normalise by max entropy
    if strategy == "margin":
        part = np.sort(proba, axis=1)
        return 1.0 - (part[:, -1] - part[:, -2])
    raise ValueError(f"unknown uncertainty strategy: {strategy!r}")


# --- Domain triggers -----------------------------------------------------------------------
def pmv_trigger(
    temperature: float,
    relative_humidity: float,
    assumptions: pmv_model.ComfortAssumptions | None = None,
) -> bool:
    """True when PMV falls outside the comfort band (|PMV| > config.PMV_NEUTRAL_BAND).

    Args:
        temperature: deg C.
        relative_humidity: percent.
        assumptions: comfort assumptions. Defaults to the fixed sedentary-indoor set. **Pass the
            same assumptions used to generate the labels** — with outdoor-driven clothing
            (DD-017) the default 0.5 clo would evaluate the trigger against a different clo than
            the labels, a silent inconsistency.
    """
    return not pmv_model.is_comfortable(
        pmv_model.pmv(temperature, relative_humidity, assumptions=assumptions)
    )


def sustained_humidity_trigger(
    df: pd.DataFrame,
    threshold: float = config.RH_SUSTAINED_THRESHOLD,
    minutes: int = config.RH_SUSTAINED_MINUTES,
) -> pd.Series:
    """Flag rows where RH has stayed above ``threshold`` for at least ``minutes`` (per zone).

    Operates on a time-ordered wide frame; returns a boolean Series aligned to ``df.index``.
    A single instantaneous reading cannot satisfy a *sustained* condition, so the dashboard uses
    the instantaneous check below; this batch version is for the loop and evaluation.
    """
    flags = pd.Series(False, index=df.index)
    for _zone, group in df.groupby(schema.ZONE):
        ordered = group.sort_values(schema.TIMESTAMP)
        above = ordered[schema.RELATIVE_HUMIDITY] > threshold
        # Duration of the current consecutive run of "above" readings, in minutes.
        run_id = (~above).cumsum()
        elapsed = (
            ordered[schema.TIMESTAMP]
            .groupby(run_id)
            .transform(lambda s: (s - s.iloc[0]).dt.total_seconds() / 60.0)
        )
        flags.loc[ordered.index] = above & (elapsed >= minutes)
    return flags


def instantaneous_humidity_trigger(
    relative_humidity: float, threshold: float = config.RH_SUSTAINED_THRESHOLD
) -> bool:
    """Single-reading RH check for the interactive dashboard (no time history available)."""
    return relative_humidity > threshold


# --- Query selection -----------------------------------------------------------------------
#: Column names carrying precomputed domain triggers (see data/sampling.py). When present in the
#: pool they are read directly; recomputing them costs ~4.3 s per call on every AL iteration.
PRECOMPUTED_TRIGGER_COLUMN = "any_triggered"

#: Query strategy that ignores the model entirely — the null hypothesis the proxy claim must beat.
RANDOM_STRATEGY = "random"


def _pool_triggers(pool: pd.DataFrame) -> np.ndarray:
    """Domain-trigger flags for a pool, reading precomputed columns when available."""
    if PRECOMPUTED_TRIGGER_COLUMN in pool.columns:
        return pool[PRECOMPUTED_TRIGGER_COLUMN].to_numpy(dtype=bool)

    humidity_flags = sustained_humidity_trigger(pool).to_numpy()
    pmv_flags = np.array(
        [
            pmv_trigger(t, rh)
            for t, rh in zip(pool[schema.TEMPERATURE], pool[schema.RELATIVE_HUMIDITY])
        ]
    )
    return humidity_flags | pmv_flags


def select_queries(
    model: ComfortModel,
    pool: pd.DataFrame,
    batch_size: int = config.QUERY_BATCH_SIZE,
    strategy: str = config.UNCERTAINTY_STRATEGY,
    use_trigger: bool = True,
    random_state: int | None = None,
) -> np.ndarray:
    """Choose the most informative pool rows to query.

    Ranks by model uncertainty, optionally prioritising rows whose domain trigger has fired so
    conservation-relevant moments (high humidity / outside the comfort band) are never skipped.

    Args:
        model: the current fitted model (unused when ``strategy`` is "random").
        pool: candidate rows. Must carry zone/timestamp unless triggers are precomputed.
        batch_size: number of rows to select.
        strategy: "entropy", "margin", or "random" (uniform selection, ignoring the model).
        use_trigger: prioritise domain-triggered rows ahead of the uncertainty ranking. On the
            Bath building the trigger fires on ~97% of rows, so this makes almost no difference
            there — measuring that is itself a finding (DD-018).
        random_state: seed for the "random" strategy.

    Returns:
        Positional indices into ``pool`` of the selected rows (length <= batch_size).
    """
    if strategy == RANDOM_STRATEGY:
        rng = np.random.default_rng(random_state)
        size = min(batch_size, len(pool))
        return rng.choice(len(pool), size=size, replace=False)

    proba = model.predict_proba(pool)
    scores = uncertainty_score(proba, strategy)

    if not use_trigger:
        return np.argsort(scores)[::-1][:batch_size]

    triggered = _pool_triggers(pool)
    # Triggered rows sort ahead of non-triggered; ties broken by uncertainty.
    ranking = np.lexsort((scores, triggered))[::-1]
    return ranking[:batch_size]


# --- The loop ------------------------------------------------------------------------------
@dataclass
class ActiveLearningResult:
    """Outcome of an active-learning run.

    ``queried_rows`` holds one record per queried instance (iteration, zone, month, T, RH, class,
    uncertainty, whether its domain trigger fired). That per-query detail is what the "in-depth
    analysis of the results" rests on — query composition by room and season, boundary behaviour,
    and the trigger's real contribution can all be recovered from it.
    """

    model: ComfortModel
    label_counts: list[int] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)
    balanced_accuracies: list[float] = field(default_factory=list)
    macro_f1s: list[float] = field(default_factory=list)
    max_uncertainties: list[float] = field(default_factory=list)
    queried_rows: list[dict] = field(default_factory=list)
    n_labels_used: int = 0


def run_active_learning(
    labelled_pool: pd.DataFrame,
    test_set: pd.DataFrame,
    *,
    seed_count: int = config.SEED_LABEL_COUNT,
    batch_size: int = config.QUERY_BATCH_SIZE,
    max_labels: int | None = None,
    random_state: int = config.RANDOM_SEED,
    strategy: str = config.UNCERTAINTY_STRATEGY,
    use_trigger: bool = True,
    record_queries: bool = False,
) -> ActiveLearningResult:
    """Run the pool-based active-learning loop against a synthetic oracle.

    The "oracle" is the pre-computed synthetic comfort class in ``labelled_pool`` — querying an
    instance simply reveals its existing label, simulating asking the occupant. Records accuracy
    on ``test_set`` after each batch so a convergence study can plot accuracy vs label budget.

    Args:
        labelled_pool: wide frame with FEATURES + the comfort-class column (the unlabelled pool;
            labels are hidden until "queried").
        test_set: held-out wide frame with the comfort-class column, for scoring.
        seed_count: initial random labels before active learning begins.
        batch_size: labels revealed per iteration.
        max_labels: stop after this many labels (defaults to the whole pool).
        random_state: seed for the initial draw, the model, and random selection.
        strategy: "entropy", "margin", or "random". "random" is the null hypothesis: if
            uncertainty sampling cannot beat it, the proxy claim fails.
        use_trigger: prioritise domain-triggered rows in query selection.
        record_queries: capture per-query detail into ``queried_rows`` for the analysis.
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score

    rng = np.random.default_rng(random_state)
    pool = labelled_pool.reset_index(drop=True)
    y_pool = pool[sl.COMFORT_CLASS].to_numpy()
    y_test = test_set[sl.COMFORT_CLASS].to_numpy()
    budget = max_labels if max_labels is not None else len(pool)

    labelled_mask = np.zeros(len(pool), dtype=bool)
    seed_idx = rng.choice(len(pool), size=min(seed_count, len(pool)), replace=False)
    labelled_mask[seed_idx] = True

    triggers = _pool_triggers(pool) if record_queries else None

    model = ComfortModel(random_state=random_state)
    result = ActiveLearningResult(model=model)
    iteration = 0

    while True:
        train = pool[labelled_mask]
        model.fit(train, y_pool[labelled_mask])

        predictions = model.predict(test_set)
        result.label_counts.append(int(labelled_mask.sum()))
        result.accuracies.append(float((predictions == y_test).mean()))
        result.balanced_accuracies.append(
            float(balanced_accuracy_score(y_test, predictions))
        )
        result.macro_f1s.append(
            float(f1_score(y_test, predictions, average="macro", zero_division=0))
        )

        remaining_idx = np.flatnonzero(~labelled_mask)
        if len(remaining_idx) == 0 or labelled_mask.sum() >= budget:
            result.max_uncertainties.append(float("nan"))
            break

        # Pool-wide peak uncertainty: the signal the convergence stop threshold watches.
        remaining_pool = pool.iloc[remaining_idx]
        if strategy == RANDOM_STRATEGY:
            result.max_uncertainties.append(float("nan"))
        else:
            scores = uncertainty_score(model.predict_proba(remaining_pool), strategy)
            result.max_uncertainties.append(float(scores.max()))

        chosen_local = select_queries(
            model,
            remaining_pool,
            batch_size,
            strategy=strategy,
            use_trigger=use_trigger,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        chosen_global = remaining_idx[chosen_local]

        if record_queries:
            for position in chosen_global:
                row = pool.iloc[position]
                result.queried_rows.append(
                    {
                        "iteration": iteration,
                        "n_labels_before": int(labelled_mask.sum()),
                        "zone": row[schema.ZONE],
                        "month": int(row[schema.TIMESTAMP].month),
                        "temperature": float(row[schema.TEMPERATURE]),
                        "relative_humidity": float(row[schema.RELATIVE_HUMIDITY]),
                        "comfort_class": row[sl.COMFORT_CLASS],
                        "triggered": bool(triggers[position]),
                    }
                )

        labelled_mask[chosen_global] = True
        iteration += 1

    result.n_labels_used = int(labelled_mask.sum())
    logger.info(
        "active learning (%s): %d labels -> accuracy %.3f",
        strategy,
        result.n_labels_used,
        result.accuracies[-1],
    )
    return result


def train_static_baseline(
    labelled: pd.DataFrame, random_state: int = config.RANDOM_SEED
) -> ComfortModel:
    """Documented fallback: train a plain Random Forest on all available labels (no AL)."""
    model = ComfortModel(random_state=random_state)
    return model.fit(labelled, labelled[sl.COMFORT_CLASS].to_numpy())


def update_with_labels(
    synthetic_prior: pd.DataFrame,
    human_labels: pd.DataFrame,
    *,
    human_weight: float = config.HUMAN_LABEL_WEIGHT,
    random_state: int = config.RANDOM_SEED,
) -> ComfortModel:
    """Retrain the comfort model on the synthetic PMV prior plus weighted human labels.

    Random Forest has no ``partial_fit``, so this *retrains* (not literally incremental) — chosen
    deliberately to keep the model tree-interpretable for SHAP. Human labels are given a higher
    sample weight so an occupant's feedback visibly personalises predictions away from the prior.

    Args:
        synthetic_prior: wide frame with FEATURES + comfort-class (the PMV-derived seed).
        human_labels: frame with FEATURES + comfort-class from real feedback (may be empty).
        human_weight: sample weight for human rows (synthetic rows are weight 1.0).

    Returns:
        A fitted ComfortModel.
    """
    feature_cols = list(config.COMFORT_VARS) + [sl.COMFORT_CLASS]
    prior = synthetic_prior[feature_cols]
    weights = np.ones(len(prior))

    if len(human_labels) > 0:
        human = human_labels[feature_cols]
        combined = pd.concat([prior, human], ignore_index=True)
        weights = np.concatenate([weights, np.full(len(human), human_weight)])
    else:
        combined = prior

    model = ComfortModel(random_state=random_state)
    return model.fit(
        combined, combined[sl.COMFORT_CLASS].to_numpy(), sample_weight=weights
    )


@dataclass(frozen=True)
class Query:
    """The single most-informative unlabelled instance to ask the occupant about."""

    index: int  # positional index into the pool
    temperature: float
    relative_humidity: float
    zone: object
    uncertainty: float
    predicted_class: str


def next_query(
    model: ComfortModel, pool: pd.DataFrame, exclude: set[int] | None = None
) -> Query | None:
    """Return the most-uncertain unlabelled pool instance to query, or None if all excluded.

    This is the "active" in active learning — the system chooses what to ask. Returns None when
    nothing remains to ask about; the caller compares ``uncertainty`` to the stop threshold.
    """
    exclude = exclude or set()
    candidates = [i for i in range(len(pool)) if i not in exclude]
    if not candidates:
        return None

    subset = pool.iloc[candidates]
    scores = uncertainty_score(model.predict_proba(subset))
    best_local = int(np.argmax(scores))
    pool_index = candidates[best_local]
    row = pool.iloc[pool_index]
    return Query(
        index=pool_index,
        temperature=float(row[schema.TEMPERATURE]),
        relative_humidity=float(row[schema.RELATIVE_HUMIDITY]),
        zone=row[schema.ZONE],
        uncertainty=float(scores[best_local]),
        predicted_class=str(model.predict(subset.iloc[[best_local]])[0]),
    )
