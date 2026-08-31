"""End-to-end dataset + model assembly, shared by the dashboard and the evaluation studies.

Single definition of "how the labelled comfort dataset is built and how the model is trained", so
the interactive app and the evaluation harness cannot drift apart.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from dissertation_code import config
from dissertation_code.comfort import clothing
from dissertation_code.comfort import synthetic_labels as sl
from dissertation_code.data import bath, schema, sources
from dissertation_code.model import active_learning as al
from dissertation_code.model import label_store, store
from dissertation_code.model.base import ComfortModel

logger = logging.getLogger(__name__)


def build_labelled_dataset(source: str = "lasdpc") -> pd.DataFrame:
    """Build the labelled wide frame for a dataset source.

    Args:
        source: "lasdpc" (the Sprint-1 slice) or "bath" (Connaught Mansions, the final dataset).

    The two sources differ in two ways that matter:

    * **Pairing.** LaSDPC logs T and RH on separate devices seconds apart, so it must be
      resampled onto a common grid before pivoting (DD-006). Bath logs both channels on the same
      row, so resampling is skipped — it would only blur real readings.
    * **Clothing.** Bath uses outdoor-driven clothing insulation and a two-class target
      (DD-017, DD-019); without them 100% of its readings label "too_cool" and the experiment
      cannot run. LaSDPC keeps the original fixed-assumption three-class behaviour.
    """
    if source == "lasdpc":
        long = sources.load_lasdpc()
        gridded = schema.resample_long(long)
        wide = schema.to_wide(gridded)
        labelled = sl.generate_labels(wide)
    elif source == "bath":
        everything = sources.load_bath_wide(include_external=True)
        is_external = everything[schema.ZONE].isin(bath.EXTERNAL_SENSORS)
        indoor = everything[~is_external]
        external = everything[is_external]

        wide = clothing.attach_clo(indoor, external)
        labelled = sl.generate_labels(
            wide,
            sl.GeneratorConfig(
                clo_column=clothing.CLOTHING_INSULATION,
                merge_warm=config.MERGE_WARM_CLASS,
            ),
        )
    else:
        raise ValueError(
            f"unknown dataset source: {source!r} (expected 'lasdpc' or 'bath')"
        )

    logger.info(
        "built labelled dataset (%s): %d rows across %d zones",
        source,
        len(labelled),
        labelled[schema.ZONE].nunique(),
    )
    return labelled


def split(
    labelled: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified random train/test split on the comfort class (reproducible).

    **Leaky on time-series data**, and deliberately retained as such: at a 5-minute cadence
    adjacent readings are near-identical, so a random split puts near-duplicates on both sides
    and inflates accuracy. Kept for the dashboard and as the documented leaky baseline — the gap
    between this and `split_temporal` quantifies the leakage. Use `split_temporal` for the
    experiment (DD-020).
    """
    train, test = train_test_split(
        labelled,
        test_size=test_size,
        random_state=random_state,
        stratify=labelled[sl.COMFORT_CLASS],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_temporal(
    labelled: pd.DataFrame,
    cutoff: str = config.TEMPORAL_SPLIT_CUTOFF,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on everything before `cutoff`, test on/after it (DD-020).

    The primary split for the experiment. It avoids the near-duplicate leakage of a random split,
    matches deployment (train on history, predict forward), and tests the seasonal transition.

    Caveat to report alongside results: train and test have different clothing assumptions and
    class balances, so an accuracy drop relative to a random split reflects distribution shift as
    well as split difficulty — the two must not be conflated.
    """
    boundary = pd.Timestamp(cutoff)
    before = labelled[schema.TIMESTAMP] < boundary
    train = labelled[before].reset_index(drop=True)
    test = labelled[~before].reset_index(drop=True)

    if train.empty or test.empty:
        raise ValueError(
            f"temporal cutoff {cutoff} leaves an empty split "
            f"(train={len(train)}, test={len(test)})"
        )

    logger.info("temporal split at %s: train=%d test=%d", cutoff, len(train), len(test))
    return train, test


def split_by_room(
    labelled: pd.DataFrame,
    holdout_rooms: tuple[str, ...] = config.HOLDOUT_ROOMS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole rooms for the cross-room generalisation arm.

    Answers a question the temporal split cannot: does a model trained on some rooms transfer to
    an unseen one? Interesting on this building because the kitchen is a genuine outlier — 66% of
    its readings exceed 75% RH, against 20% or less elsewhere.
    """
    is_holdout = labelled[schema.ZONE].isin(holdout_rooms)
    train = labelled[~is_holdout].reset_index(drop=True)
    test = labelled[is_holdout].reset_index(drop=True)

    if train.empty or test.empty:
        raise ValueError(
            f"holdout rooms {holdout_rooms} leave an empty split "
            f"(train={len(train)}, test={len(test)})"
        )

    logger.info(
        "room split holding out %s: train=%d test=%d",
        holdout_rooms,
        len(train),
        len(test),
    )
    return train, test


def train_static_model() -> tuple[ComfortModel, pd.DataFrame, pd.DataFrame]:
    """Build the dataset and train the documented static-RF baseline; return model + splits."""
    labelled = build_labelled_dataset()
    train, test = split(labelled)
    model = al.train_static_baseline(train)
    return model, train, test


# --- Live closed-loop lifecycle (active-learning items 3-5) --------------------------------
def retrain_live_model(synthetic_prior: pd.DataFrame) -> ComfortModel:
    """Retrain the live model on the synthetic prior + all stored human labels, and persist it."""
    human = label_store.load_labels()
    model = al.update_with_labels(synthetic_prior, human)
    store.save_model(
        model,
        manifest={"n_synthetic": len(synthetic_prior), "n_human_labels": len(human)},
    )
    logger.info(
        "retrained live model on %d synthetic + %d human labels",
        len(synthetic_prior),
        len(human),
    )
    return model


def load_or_train_live_model(
    synthetic_prior: pd.DataFrame,
) -> ComfortModel:
    """Load the persisted live model if present, otherwise train one from the prior + labels."""
    model = store.load_model()
    if model is not None:
        return model
    return retrain_live_model(synthetic_prior)
