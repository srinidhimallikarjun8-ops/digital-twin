"""Robustness studies: does the conclusion survive its own assumptions?

Architecture Layer 7. Two studies that exist to attack the experiment rather than support it.

**Clothing sensitivity.** The outdoor-driven clothing rule (DD-017) is a modelling choice, not a
measurement. Its parameters were picked to be physically plausible, but a result that depends on
that particular rule would be an artefact of the choice. `clothing_sensitivity` re-generates the
labels under several defensible clothing assumptions and reports how the class balance and model
performance move.

**Circularity.** Clothing is derived from outdoor temperature, and clothing feeds PMV, which
generates the label. If the classifier could also see outdoor temperature (or a proxy such as the
date), it could partly learn the label-generation rule rather than the comfort relationship.

The model here takes **only indoor temperature and relative humidity** — outdoor temperature is
never a feature (enforced by `ComfortModel._matrix`). So the direct path is closed by construction.
A weaker path remains: indoor temperature correlates with outdoor temperature, so seasonal
information reaches the model indirectly. `circularity_check` quantifies how much of the label is
explained by that seasonal component alone, by comparing the real model against a
season-only baseline that sees *nothing but* the month.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from dissertation_code import config
from dissertation_code.comfort import clothing
from dissertation_code.comfort import synthetic_labels as sl
from dissertation_code.data import sampling, schema
from dissertation_code.evaluation import validation
from dissertation_code.model import active_learning as al

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClothingVariant:
    """One clothing assumption to test the conclusion against."""

    name: str
    description: str
    #: None means "use the outdoor-driven rule with these parameters"; a float means fixed clo.
    fixed_clo: float | None = None
    clo_min: float = config.CLO_MIN
    clo_max: float = config.CLO_MAX
    slope: float = config.CLO_SLOPE
    reference_temperature: float = config.CLO_REF_TEMP


#: Defensible alternatives spanning fixed and adaptive assumptions, plus a deliberately wrong one.
CLOTHING_VARIANTS: tuple[ClothingVariant, ...] = (
    ClothingVariant(
        "adopted",
        "Outdoor-driven, adopted parameters (DD-017)",
    ),
    ClothingVariant(
        "fixed_1.0",
        "Fixed 1.0 clo — a business suit, ISO 7730's winter reference",
        fixed_clo=1.0,
    ),
    ClothingVariant(
        "fixed_1.25",
        "Fixed 1.25 clo — suit plus a jumper",
        fixed_clo=1.25,
    ),
    ClothingVariant(
        "shallow_slope",
        "Outdoor-driven but half as responsive to outdoor temperature",
        slope=config.CLO_SLOPE / 2,
    ),
    ClothingVariant(
        "steep_slope",
        "Outdoor-driven but twice as responsive",
        slope=config.CLO_SLOPE * 2,
    ),
    ClothingVariant(
        "narrow_range",
        "Outdoor-driven, but occupants vary clothing only within 0.8-1.1 clo",
        clo_min=0.8,
        clo_max=1.1,
    ),
    ClothingVariant(
        "fixed_0.5_inherited",
        "Fixed 0.5 clo — the inherited LaSDPC assumption; expected to be degenerate",
        fixed_clo=0.5,
    ),
)


def relabel_with_clothing(
    wide_with_clo: pd.DataFrame, variant: ClothingVariant
) -> pd.DataFrame:
    """Regenerate synthetic labels under an alternative clothing assumption.

    Args:
        wide_with_clo: indoor frame that already carries an outdoor-driven clo column, plus the
            outdoor running mean used to derive it (see `clothing.attach_clo`).
        variant: the assumption to apply.

    Returns:
        A relabelled copy — same readings, different labels.
    """
    frame = wide_with_clo.copy()

    if variant.fixed_clo is not None:
        frame[clothing.CLOTHING_INSULATION] = variant.fixed_clo
    else:
        # Recover the outdoor running mean by inverting the adopted mapping, then re-apply the
        # variant's parameters. Values that were clipped cannot be recovered exactly, so this is
        # only valid for variants that widen or match the adopted range.
        outdoor = (
            config.CLO_REF_TEMP
            + (config.CLO_MAX - frame[clothing.CLOTHING_INSULATION]) / config.CLO_SLOPE
        )
        frame[clothing.CLOTHING_INSULATION] = clothing.clo_from_outdoor(
            outdoor.to_numpy(),
            clo_min=variant.clo_min,
            clo_max=variant.clo_max,
            slope=variant.slope,
            reference_temperature=variant.reference_temperature,
        )

    labelled = sl.generate_labels(
        frame.drop(
            columns=[
                c
                for c in (
                    sl.PMV_VALUE,
                    sl.SENSATION_VOTE,
                    sl.COMFORT_LABEL,
                    sl.COMFORT_CLASS,
                )
                if c in frame.columns
            ]
        ),
        sl.GeneratorConfig(
            clo_column=clothing.CLOTHING_INSULATION,
            merge_warm=config.MERGE_WARM_CLASS,
        ),
    )
    return labelled


def clothing_sensitivity(
    wide_with_clo: pd.DataFrame,
    variants: tuple[ClothingVariant, ...] = CLOTHING_VARIANTS,
    pool_size: int = 5_000,
    seeds: tuple[int, ...] = (42, 43, 44),
) -> pd.DataFrame:
    """Re-run the pipeline under each clothing assumption and report how conclusions move.

    The question is not "which clothing rule is right" — none can be verified without occupant
    observation — but "does the finding depend on the choice". A conclusion that only holds for
    the adopted parameters would be an artefact.

    Returns one row per (variant, seed) with class balance, accuracy and its estimated maximum.
    """
    from dissertation_code import pipeline

    rows = []
    for variant in variants:
        labelled = relabel_with_clothing(wide_with_clo, variant)
        balance = labelled[sl.COMFORT_CLASS].value_counts(normalize=True)
        n_classes = labelled[sl.COMFORT_CLASS].nunique()

        if n_classes < 2:
            # Degenerate labelling: no decision boundary exists, so no model can be trained.
            rows.append(
                {
                    "variant": variant.name,
                    "description": variant.description,
                    "seed": None,
                    "n_classes": n_classes,
                    "too_cool_share": float(
                        balance.get(config.COMFORT_CLASS_TOO_COOL, 0.0)
                    ),
                    "mean_clo": float(labelled[clothing.CLOTHING_INSULATION].mean()),
                    "accuracy": float("nan"),
                    "estimated_max_accuracy": float("nan"),
                    "ceiling_ratio": float("nan"),
                    "degenerate": True,
                }
            )
            logger.warning(
                "variant %s produced a single class — no experiment possible",
                variant.name,
            )
            continue

        train, test = pipeline.split_temporal(labelled)
        test_sample = test.sample(
            n=min(len(test), config.TEST_SET_SIZE), random_state=config.RANDOM_SEED
        )

        for seed in seeds:
            pool = sampling.build_pool(train, n=pool_size, random_state=seed)
            model = al.train_static_baseline(pool, random_state=seed)
            report = validation.validate(model, test_sample)
            rows.append(
                {
                    "variant": variant.name,
                    "description": variant.description,
                    "seed": seed,
                    "n_classes": n_classes,
                    "too_cool_share": float(
                        balance.get(config.COMFORT_CLASS_TOO_COOL, 0.0)
                    ),
                    "mean_clo": float(labelled[clothing.CLOTHING_INSULATION].mean()),
                    "accuracy": report.accuracy,
                    "estimated_max_accuracy": report.estimated_max_accuracy,
                    "ceiling_ratio": report.ceiling_ratio,
                    "degenerate": False,
                }
            )

    return pd.DataFrame(rows)


def circularity_check(
    labelled: pd.DataFrame,
    pool_size: int = 5_000,
    seeds: tuple[int, ...] = (42, 43, 44),
) -> pd.DataFrame:
    """How much of the label is explained by seasonal information alone?

    Clothing is derived from outdoor temperature, so the label carries a seasonal component. The
    model never sees outdoor temperature (only indoor T + RH), but indoor temperature correlates
    with outdoor, so season leaks in indirectly.

    Three arms make the size of that leak visible:

    * ``month_only`` — predicts from the calendar month alone, seeing no thermal reading at all.
      This is the pure label-generation-rule baseline: whatever it achieves is attributable to
      seasonal structure rather than to comfort physics.
    * ``humidity_only`` — sees relative humidity but not temperature. PMV depends far more on
      temperature, so this arm should be close to the majority-class rate.
    * ``full`` — the real model: indoor temperature + relative humidity.

    If ``month_only`` approaches ``full``, the model is largely reproducing the labelling rule and
    the result must be reported as such. A large gap indicates the model is using thermal
    information the seasonal component does not supply.
    """
    from sklearn.ensemble import RandomForestClassifier

    from dissertation_code import pipeline

    train, test = pipeline.split_temporal(labelled)
    test_sample = sampling.attach_month(
        test.sample(
            n=min(len(test), config.TEST_SET_SIZE), random_state=config.RANDOM_SEED
        )
    )
    y_test = test_sample[sl.COMFORT_CLASS].to_numpy()
    majority_rate = float(pd.Series(y_test).value_counts(normalize=True).iloc[0])

    arms = {
        "month_only": [sampling.MONTH],
        "humidity_only": [schema.RELATIVE_HUMIDITY],
        "temperature_only": [schema.TEMPERATURE],
        "full": [schema.TEMPERATURE, schema.RELATIVE_HUMIDITY],
    }

    rows = []
    for seed in seeds:
        pool = sampling.build_pool(train, n=pool_size, random_state=seed)
        y_pool = pool[sl.COMFORT_CLASS].to_numpy()

        for arm, features in arms.items():
            estimator = RandomForestClassifier(
                n_estimators=config.N_ESTIMATORS,
                random_state=seed,
                class_weight="balanced",
            )
            estimator.fit(pool[features], y_pool)
            predictions = estimator.predict(test_sample[features])
            n_correct = int((predictions == y_test).sum())
            accuracy = n_correct / len(y_test)
            low, high = validation.wilson_interval(n_correct, len(y_test))

            rows.append(
                {
                    "arm": arm,
                    "features": ", ".join(features),
                    "seed": seed,
                    "accuracy": accuracy,
                    "ci_lower": low,
                    "ci_upper": high,
                    "majority_class_rate": majority_rate,
                    "lift_over_majority": accuracy - majority_rate,
                }
            )

    return pd.DataFrame(rows)


def summarise_circularity(result: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- sd accuracy per arm, with each arm's share of the full model's lift.

    ``share_of_full_lift`` is the interpretable number: the fraction of the full model's
    improvement over the majority-class baseline that this arm reproduces on its own.
    """
    summary = (
        result.groupby(["arm", "features"])["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    majority = float(result["majority_class_rate"].iloc[0])
    summary["majority_class_rate"] = majority
    summary["lift_over_majority"] = summary["mean"] - majority

    full_lift = summary.loc[summary["arm"] == "full", "lift_over_majority"]
    if not full_lift.empty and abs(full_lift.iloc[0]) > 1e-9:
        summary["share_of_full_lift"] = (
            summary["lift_over_majority"] / full_lift.iloc[0]
        )

    return summary.sort_values("mean", ascending=False).reset_index(drop=True)
