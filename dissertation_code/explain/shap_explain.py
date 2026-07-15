"""SHAP feature attributions for a comfort prediction (architecture Layer 4).

Caveat (kept visible deliberately): temperature and relative humidity are correlated, so SHAP can
attribute importance to one feature that is partly a proxy for the other. The numeric attribution
is therefore stored in the audit log as supporting evidence, not presented to the user as the sole
truth; the plain-language sentence (narrate.py) is what the user sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from dissertation_code.model.base import FEATURES, ComfortModel


@dataclass(frozen=True)
class Attribution:
    """SHAP attribution for one prediction: per-feature contribution toward the predicted class."""

    predicted_class: str
    contributions: dict[
        str, float
    ]  # feature -> signed SHAP value (toward predicted_class)
    feature_values: dict[str, float]  # feature -> the input value

    def ranked(self) -> list[tuple[str, float]]:
        """Features ordered by absolute contribution, most influential first."""
        return sorted(
            self.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True
        )


def explain_instance(model: ComfortModel, instance: pd.DataFrame) -> Attribution:
    """Compute the SHAP attribution for a single-row instance toward its predicted class.

    Args:
        model: a fitted ComfortModel.
        instance: a one-row frame containing the FEATURES columns.

    Returns:
        An Attribution with signed per-feature contributions toward the predicted class.
    """
    if len(instance) != 1:
        raise ValueError("explain_instance expects exactly one row")

    predicted_class = str(model.predict(instance)[0])
    class_index = list(model.classes_).index(predicted_class)

    explainer = shap.TreeExplainer(model.estimator)
    values = explainer.shap_values(instance[FEATURES])
    per_class = _values_for_class(values, class_index, n_features=len(FEATURES))

    contributions = {feat: float(per_class[i]) for i, feat in enumerate(FEATURES)}
    feature_values = {feat: float(instance.iloc[0][feat]) for feat in FEATURES}
    return Attribution(predicted_class, contributions, feature_values)


def _values_for_class(values, class_index: int, n_features: int) -> np.ndarray:
    """Extract the single-row, single-class SHAP vector across SHAP's output shapes.

    Different SHAP / sklearn versions return either a list (one array per class) or a single
    array shaped (n_samples, n_features, n_classes). This normalises both to a length-n_features
    vector for the requested class.
    """
    if isinstance(values, list):  # list[class] of (n_samples, n_features)
        return np.asarray(values[class_index])[0]
    arr = np.asarray(values)
    if arr.ndim == 3:  # (n_samples, n_features, n_classes)
        return arr[0, :, class_index]
    return arr[0]  # (n_samples, n_features)
