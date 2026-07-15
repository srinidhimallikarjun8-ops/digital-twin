"""End-to-end dataset + model assembly, shared by the dashboard and the evaluation studies.

Single definition of "how the labelled comfort dataset is built and how the model is trained", so
the interactive app and the evaluation harness cannot drift apart.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from dissertation_code import config
from dissertation_code.comfort import synthetic_labels as sl
from dissertation_code.data import schema, sources
from dissertation_code.model import active_learning as al
from dissertation_code.model import label_store, store
from dissertation_code.model.base import ComfortModel

logger = logging.getLogger(__name__)


def build_labelled_dataset() -> pd.DataFrame:
    """Load LaSDPC, resample, pair T+RH, and attach synthetic comfort labels (wide frame)."""
    long = sources.load_lasdpc()
    gridded = schema.resample_long(long)
    wide = schema.to_wide(gridded)
    labelled = sl.generate_labels(wide)
    logger.info(
        "built labelled dataset: %d rows across %d zones",
        len(labelled),
        labelled[schema.ZONE].nunique(),
    )
    return labelled


def split(
    labelled: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on the comfort class (reproducible)."""
    train, test = train_test_split(
        labelled,
        test_size=test_size,
        random_state=random_state,
        stratify=labelled[sl.COMFORT_CLASS],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


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
