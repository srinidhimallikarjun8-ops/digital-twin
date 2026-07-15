"""Active-learning convergence study (architecture Layer 7; Sprint 4 / O5).

Records model accuracy as a function of label budget and reports whether active learning reaches
within ``CONVERGENCE_ACCURACY_TOLERANCE`` of the full-labelled baseline using no more than
``LABEL_BUDGET_FRACTION`` of the labels — the methodology's stated convergence target.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dissertation_code import config
from dissertation_code.comfort import synthetic_labels as sl
from dissertation_code.model import active_learning as al


@dataclass(frozen=True)
class ConvergenceReport:
    label_counts: list[int]
    accuracies: list[float]
    baseline_accuracy: float
    labels_to_target: (
        int | None
    )  # labels needed to reach (baseline - tolerance); None if never
    total_pool: int

    @property
    def fraction_to_target(self) -> float | None:
        if self.labels_to_target is None:
            return None
        return self.labels_to_target / self.total_pool

    @property
    def meets_target(self) -> bool:
        frac = self.fraction_to_target
        return frac is not None and frac <= config.LABEL_BUDGET_FRACTION


def run_convergence_study(
    pool: pd.DataFrame,
    test_set: pd.DataFrame,
    tolerance: float = config.CONVERGENCE_ACCURACY_TOLERANCE,
) -> ConvergenceReport:
    """Run the active-learning loop and compare its label-efficiency to the full-labelled baseline.

    Args:
        pool: labelled wide frame used as the active-learning pool.
        test_set: held-out wide frame for scoring.
        tolerance: how close to the baseline counts as "converged".
    """
    baseline = al.train_static_baseline(pool)
    y_test = test_set[sl.COMFORT_CLASS].to_numpy()
    baseline_accuracy = float((baseline.predict(test_set) == y_test).mean())

    result = al.run_active_learning(pool, test_set)
    target = baseline_accuracy - tolerance

    labels_to_target: int | None = None
    for n_labels, acc in zip(result.label_counts, result.accuracies):
        if acc >= target:
            labels_to_target = n_labels
            break

    return ConvergenceReport(
        label_counts=result.label_counts,
        accuracies=result.accuracies,
        baseline_accuracy=baseline_accuracy,
        labels_to_target=labels_to_target,
        total_pool=len(pool),
    )
