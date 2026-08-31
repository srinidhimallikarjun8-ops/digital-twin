"""Validation of the trained comfort model (architecture Layer 7).

Raw held-out accuracy is close to meaningless on this dataset, and reporting it alone would be
misleading. The synthetic labels are ``round(PMV + N(0, sigma))``, so the label itself is
stochastic: two identical (T, RH) readings can carry different classes. That imposes a hard
**Bayes ceiling** — the accuracy a perfect model would achieve — which on the Bath pool sits
near 0.60, not 1.0.

Measured example: a model scoring 0.563 on the temporal split is achieving ~95% of the 0.595
ceiling. Quoted bare, 0.563 reads as a poor model; against the ceiling it is close to optimal.
Every accuracy figure in the dissertation should therefore be accompanied by its ceiling-relative
counterpart.

The other checks here target the *mechanism* rather than the score: uncertainty sampling depends
on calibrated probabilities, so if the model's confidence is systematically wrong then active
learning is selecting on a distorted signal — a failure that accuracy alone would never reveal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from dissertation_code import config
from dissertation_code.comfort import synthetic_labels as sl
from dissertation_code.model.base import ComfortModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationReport:
    """Held-out performance, interpreted against the synthetic-label noise ceiling.

    Accuracy is reported with its sample size and a Wilson 95% interval, and alongside the
    majority-class rate. Quoting accuracy alone would hide the two failure modes that matter on
    an imbalanced two-class problem: a model that always predicts the majority class, and a model
    that looks adequate overall while performing poorly on the minority class.
    """

    n_samples: int
    accuracy: float
    accuracy_ci: tuple[float, float]  # Wilson 95% interval
    majority_class_rate: float  # accuracy of always predicting the commonest class
    balanced_accuracy: float
    macro_f1: float
    per_class: pd.DataFrame  # precision / recall / F1 / support, per class
    estimated_max_accuracy: float  # under the assumed label-generation process
    ceiling_ratio: float
    brier_score: float
    confusion: pd.DataFrame
    class_balance: dict[str, float]
    pmv_agreement: float | None

    def summary_lines(self) -> list[str]:
        low, high = self.accuracy_ci
        lines = [
            f"n (test)              {self.n_samples}",
            f"accuracy              {self.accuracy:.4f}  95% CI [{low:.4f}, {high:.4f}]",
            f"majority-class rate   {self.majority_class_rate:.4f}  (trivial baseline)",
            f"balanced accuracy     {self.balanced_accuracy:.4f}",
            f"macro F1              {self.macro_f1:.4f}",
            (
                f"est. max accuracy     {self.estimated_max_accuracy:.4f}  "
                "(under the assumed label-generation + noise process, given these features)"
            ),
            f"ratio to est. max     {self.ceiling_ratio:.3f}",
            f"Brier score           {self.brier_score:.4f}  (lower is better-calibrated)",
        ]
        if self.pmv_agreement is not None:
            lines.append(
                f"PMV agreement         {self.pmv_agreement:.3f}  (expect ~0.33, Cheung 2019)"
            )
        return lines


def wilson_interval(
    successes: int, trials: int, z: float = 1.96
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and behaves sensibly
    for proportions near the bounds and for small samples.
    """
    if trials == 0:
        return (float("nan"), float("nan"))
    phat = successes / trials
    denominator = 1 + z**2 / trials
    centre = phat + z**2 / (2 * trials)
    spread = z * np.sqrt(phat * (1 - phat) / trials + z**2 / (4 * trials**2))
    return ((centre - spread) / denominator, (centre + spread) / denominator)


def estimated_max_accuracy(
    pmv_values: np.ndarray,
    noise_sigma: float = config.NOISE_STD,
    vote_threshold: int = config.COMFORT_VOTE_THRESHOLD,
    merge_warm: bool = config.MERGE_WARM_CLASS,
) -> float:
    """Estimated maximum expected accuracy **under this study's label-generation process**.

    This is not a universal bound. It is the Bayes-optimal accuracy *conditional on the
    simulation's own assumptions*, and it should always be reported with them attached.

    **Derivation.** A synthetic label is generated as::

        vote  = clip(round(PMV + e), -3, +3),   e ~ N(0, sigma)
        class = too_cool     if vote <  -vote_threshold
                too_warm     if vote >  +vote_threshold   (unless merged)
                comfortable  otherwise

    Because ``round`` sends everything in ``(k - 0.5, k + 0.5]`` to ``k``, the class boundaries in
    the *continuous* variable ``PMV + e`` sit at ``+/-(vote_threshold + 0.5)``. So for a row with
    a given PMV::

        P(too_cool)    = Phi( -(t + 0.5); PMV, sigma )
        P(too_warm)    = 1 - Phi( +(t + 0.5); PMV, sigma )
        P(comfortable) = 1 - P(too_cool) - P(too_warm)

    A predictor that sees only (temperature, humidity) sees PMV at best — the noise ``e`` is by
    construction unpredictable from the features. The best achievable rule is therefore to predict
    the most probable class at each PMV, scoring ``max_c P(c | PMV)``. Averaging over the
    evaluation set gives the estimate returned here.

    **Assumptions this figure depends on** (change any and the number changes):

    1. Labels really are PMV + additive Gaussian noise (the simulation's design, not a fact about
       occupants).
    2. ``sigma`` is the value used to generate the labels being scored.
    3. The features carry no information about the noise draw ``e``.
    4. The class mapping (``vote_threshold``, ``merge_warm``) matches the labels being scored.
    5. PMV itself is computed with the same assumptions used at labelling time — notably the same
       clothing insulation (DD-017).

    Under (3), this is an upper bound for *any* model on these features, not just this one. It is
    silent about how well a model would predict **real occupant comfort**, for which no ground
    truth exists in this dataset.

    Args:
        pmv_values: the underlying (noise-free) PMV of each evaluation row.
        noise_sigma: sigma of the label noise.
        vote_threshold: |vote| <= this is "comfortable".
        merge_warm: whether the warm class was merged into "comfortable" (DD-019).

    Returns:
        Mean probability of the most likely class under the assumed process.
    """
    pmv_values = np.asarray(pmv_values, dtype=float)
    # Votes are rounded, so class boundaries sit at +/-(threshold + 0.5).
    lower = -(vote_threshold + 0.5)
    upper = vote_threshold + 0.5

    p_too_cool = norm.cdf(lower, loc=pmv_values, scale=noise_sigma)
    p_too_warm = 1.0 - norm.cdf(upper, loc=pmv_values, scale=noise_sigma)
    p_comfortable = 1.0 - p_too_cool - p_too_warm

    if merge_warm:
        class_probabilities = np.vstack([p_too_cool, p_comfortable + p_too_warm])
    else:
        class_probabilities = np.vstack([p_too_cool, p_comfortable, p_too_warm])

    return float(class_probabilities.max(axis=0).mean())


@dataclass(frozen=True)
class LabelSignalReport:
    """Whether the label-generation process leaves any learnable signal at all.

    This diagnostic decides whether a supervised experiment on these labels is meaningful
    *before* any model is trained. It is separate from model performance: if the estimated
    maximum accuracy is no better than always predicting the commonest class, then no model,
    however good, can demonstrate anything — and a null active-learning result would say nothing
    about active learning.
    """

    pmv_std: float  # spread of the underlying signal
    noise_sigma: float  # spread of the label noise
    signal_to_noise: float  # pmv_std / noise_sigma
    pmv_min: float
    pmv_max: float
    boundary_percentile: float  # where the class boundary falls in the PMV distribution
    estimated_max_accuracy: float
    majority_class_rate: float
    headroom: float  # estimated_max_accuracy - majority_class_rate
    pmv_agreement: float  # for the Cheung et al. (2019) calibration check

    @property
    def is_learnable(self) -> bool:
        """True when the optimal rule beats the trivial baseline by a usable margin.

        The 0.05 threshold is a judgement, not a standard: below it, the best possible model is
        within a few percentage points of "always predict the majority class", so differences
        between query strategies would be indistinguishable from noise at any realistic
        label budget.
        """
        return self.headroom >= 0.05

    def summary_lines(self) -> list[str]:
        return [
            f"PMV range              [{self.pmv_min:.2f}, {self.pmv_max:.2f}]",
            f"PMV std dev (signal)   {self.pmv_std:.3f}",
            f"label noise sigma      {self.noise_sigma:.3f}",
            f"signal-to-noise        {self.signal_to_noise:.3f}",
            f"boundary percentile    {self.boundary_percentile:.1f}%",
            f"PMV-vote agreement     {self.pmv_agreement:.3f}  (Cheung 2019 target ~0.33)",
            f"est. max accuracy      {self.estimated_max_accuracy:.4f}",
            f"majority-class rate    {self.majority_class_rate:.4f}",
            f"headroom               {self.headroom:+.4f}",
            f"learnable              {self.is_learnable}",
        ]


def label_signal_report(
    labelled: pd.DataFrame,
    noise_sigma: float = config.NOISE_STD,
    vote_threshold: int = config.COMFORT_VOTE_THRESHOLD,
) -> LabelSignalReport:
    """Diagnose how much learnable signal the synthetic labels retain.

    Motivation: a supervised experiment is only meaningful if the labels are predictable from the
    features better than chance. Synthetic labels are ``round(PMV + N(0, sigma))``, so the
    achievable signal depends on the *ratio* between the spread of PMV across the dataset and
    sigma — not on sigma alone.

    On a dataset whose PMV varies widely, sigma=1.0 leaves plenty of structure. On one where PMV
    barely varies, the same sigma erases it. The same noise parameter can therefore be
    well-calibrated against the literature and still destroy learnability, which is exactly the
    situation on this building and the reason this diagnostic exists.

    Args:
        labelled: frame carrying the PMV column and the comfort-class column.
        noise_sigma: sigma used to generate the labels.
        vote_threshold: |vote| <= this is "comfortable".
    """
    pmv_values = labelled[sl.PMV_VALUE].to_numpy()
    class_counts = labelled[sl.COMFORT_CLASS].value_counts(normalize=True)

    boundary = -(vote_threshold + 0.5)
    ceiling = estimated_max_accuracy(
        pmv_values, noise_sigma=noise_sigma, vote_threshold=vote_threshold
    )
    majority = float(class_counts.iloc[0])

    return LabelSignalReport(
        pmv_std=float(pmv_values.std()),
        noise_sigma=noise_sigma,
        signal_to_noise=float(pmv_values.std() / noise_sigma),
        pmv_min=float(pmv_values.min()),
        pmv_max=float(pmv_values.max()),
        boundary_percentile=float(100.0 * (pmv_values < boundary).mean()),
        estimated_max_accuracy=ceiling,
        majority_class_rate=majority,
        headroom=ceiling - majority,
        pmv_agreement=sl.pmv_agreement_rate(labelled),
    )


def validate(
    model: ComfortModel,
    test_set: pd.DataFrame,
    positive_class: str = config.COMFORT_CLASS_TOO_COOL,
) -> ValidationReport:
    """Score a fitted model on a held-out set.

    Reports per-class precision/recall/F1 and the majority-class baseline alongside accuracy, so
    a model that simply predicts the commonest class cannot look successful.

    Args:
        model: fitted ComfortModel.
        test_set: held-out frame carrying FEATURES and the comfort-class column. When it also
            carries the underlying PMV, the estimated maximum accuracy is computed from it.
        positive_class: class treated as positive for the Brier calibration score.
    """
    y_true = test_set[sl.COMFORT_CLASS].to_numpy()
    predictions = model.predict(test_set)
    proba = model.predict_proba(test_set)

    labels = list(model.classes_)
    confusion = pd.DataFrame(
        confusion_matrix(y_true, predictions, labels=labels),
        index=pd.Index(labels, name="actual"),
        columns=pd.Index(labels, name="predicted"),
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=labels, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "class": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )

    if positive_class in labels:
        positive_column = labels.index(positive_class)
        brier = float(
            brier_score_loss(
                (y_true == positive_class).astype(int), proba[:, positive_column]
            )
        )
    else:  # pragma: no cover - only if the model never saw the positive class
        brier = float("nan")

    n_correct = int((predictions == y_true).sum())
    n_samples = len(y_true)
    accuracy = n_correct / n_samples
    class_counts = pd.Series(y_true).value_counts()
    majority_rate = float(class_counts.iloc[0] / n_samples)

    ceiling = (
        estimated_max_accuracy(test_set[sl.PMV_VALUE].to_numpy())
        if sl.PMV_VALUE in test_set.columns
        else float("nan")
    )

    report = ValidationReport(
        n_samples=n_samples,
        accuracy=accuracy,
        accuracy_ci=wilson_interval(n_correct, n_samples),
        majority_class_rate=majority_rate,
        balanced_accuracy=float(balanced_accuracy_score(y_true, predictions)),
        macro_f1=float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        per_class=per_class,
        estimated_max_accuracy=ceiling,
        ceiling_ratio=(
            float(accuracy / ceiling) if not np.isnan(ceiling) else float("nan")
        ),
        brier_score=brier,
        confusion=confusion,
        class_balance=test_set[sl.COMFORT_CLASS]
        .value_counts(normalize=True)
        .round(4)
        .to_dict(),
        pmv_agreement=(
            sl.pmv_agreement_rate(test_set)
            if {sl.PMV_VALUE, sl.SENSATION_VOTE}.issubset(test_set.columns)
            else None
        ),
    )

    for line in report.summary_lines():
        logger.info("  %s", line)
    return report


def calibration_curve(
    model: ComfortModel,
    test_set: pd.DataFrame,
    positive_class: str = config.COMFORT_CLASS_TOO_COOL,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Predicted probability vs observed frequency, binned.

    Active learning selects on predicted probabilities, so systematic miscalibration means the
    query strategy is ranking on a distorted signal. A well-calibrated model has
    ``observed_frequency`` tracking ``mean_predicted`` along the diagonal.
    """
    labels = list(model.classes_)
    if positive_class not in labels:  # pragma: no cover
        raise ValueError(f"{positive_class!r} not among model classes {labels}")

    proba = model.predict_proba(test_set)[:, labels.index(positive_class)]
    actual = (test_set[sl.COMFORT_CLASS].to_numpy() == positive_class).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(proba, bins) - 1, 0, n_bins - 1)

    rows = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        rows.append(
            {
                "bin": bin_id,
                "bin_lower": bins[bin_id],
                "bin_upper": bins[bin_id + 1],
                "mean_predicted": float(proba[mask].mean()),
                "observed_frequency": float(actual[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def risk_coverage(
    model: ComfortModel,
    test_set: pd.DataFrame,
    positive_class: str = config.COMFORT_CLASS_TOO_COOL,
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
) -> pd.DataFrame:
    """Precision and coverage as a confidence threshold is raised (selective prediction).

    Forcing a prediction on every reading is the wrong evaluation for this system. A heritage
    building only warrants intervention when discomfort is *confidently* identified — the Burra
    Charter minimum-intervention principle. A model that declines to act on ambiguous readings is
    behaving correctly, not failing.

    Standard accuracy penalises exactly that behaviour: it grades the model on readings where the
    label is a coin flip. This function instead reports, at each confidence threshold, how often
    the model acts (`coverage`) and how often it is right when it does (`precision`).

    A system that never reaches a usable threshold is itself a finding: under the labelling process
    in use, no intervention is confidently warranted.

    Args:
        model: fitted ComfortModel.
        test_set: held-out frame with FEATURES and the comfort-class column.
        positive_class: the class whose detection would trigger an action.
        thresholds: confidence levels to report.

    Returns:
        One row per threshold: coverage, precision, recall, and the count acted on.
    """
    labels = list(model.classes_)
    if positive_class not in labels:  # pragma: no cover
        raise ValueError(f"{positive_class!r} not among model classes {labels}")

    proba = model.predict_proba(test_set)[:, labels.index(positive_class)]
    actual = (test_set[sl.COMFORT_CLASS].to_numpy() == positive_class).astype(int)
    n_positive = int(actual.sum())
    base_rate = float(actual.mean())

    rows = []
    for threshold in thresholds:
        acted = proba >= threshold
        n_acted = int(acted.sum())
        rows.append(
            {
                "threshold": threshold,
                "n_acted": n_acted,
                "coverage": n_acted / len(actual),
                "precision": float(actual[acted].mean()) if n_acted else float("nan"),
                "recall": (
                    float(actual[acted].sum() / n_positive)
                    if n_positive
                    else float("nan")
                ),
                "base_rate": base_rate,
                "lift_over_base": (
                    float(actual[acted].mean() - base_rate) if n_acted else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)
