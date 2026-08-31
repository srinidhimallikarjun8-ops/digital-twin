"""Interpretable base comfort model (architecture Layer 3).

A thin wrapper around scikit-learn's RandomForestClassifier, fixed to the two permitted model
inputs (temperature + relative humidity) and the three directional comfort classes. Random Forest
is chosen for interpretability (SHAP/TreeExplainer, feature importances) and because it needs no
large labelled set — see the method comparison (detailed_dissertation.md Table 2.1) and DD-001.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from dissertation_code import config
from dissertation_code.data import schema

FEATURES = list(config.COMFORT_VARS)  # ["temperature", "relative_humidity"]


class ComfortModel:
    """Random-forest comfort classifier over temperature + relative humidity."""

    def __init__(
        self,
        n_estimators: int = config.N_ESTIMATORS,
        random_state: int = config.RANDOM_SEED,
    ) -> None:
        self._clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def classes_(self) -> np.ndarray:
        """Class labels in the order used by ``predict_proba`` columns."""
        return self._clf.classes_

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> ComfortModel:
        """Fit on a frame containing the FEATURES columns and a comfort-class target.

        ``sample_weight`` lets human labels outweigh the synthetic PMV prior during retraining.
        """
        self._clf.fit(self._matrix(X), y, sample_weight=sample_weight)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the comfort class for each row."""
        return self._clf.predict(self._matrix(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Class-probability matrix (rows = samples, columns = ``classes_``)."""
        return self._clf.predict_proba(self._matrix(X))

    @property
    def estimator(self) -> RandomForestClassifier:
        """The underlying sklearn estimator (e.g. for SHAP's TreeExplainer)."""
        return self._clf

    @staticmethod
    def _matrix(X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"input is missing model features {missing}")
        # Select ONLY the permitted features — defends the T+RH-only scope at the model boundary.
        return X[FEATURES]


def make_instance(temperature: float, relative_humidity: float) -> pd.DataFrame:
    """Build a single-row feature frame from raw T+RH values (used by the dashboard)."""
    return pd.DataFrame(
        {
            schema.TEMPERATURE: [temperature],
            schema.RELATIVE_HUMIDITY: [relative_humidity],
        }
    )
