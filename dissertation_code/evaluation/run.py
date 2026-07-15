"""Run the Sprint 4 evaluation studies end-to-end (scenario testing + convergence).

Usage: ``uv run python -m dissertation_code.evaluation.run``
"""

from __future__ import annotations

import logging

from dissertation_code import config, pipeline
from dissertation_code.evaluation import convergence, scenarios
from dissertation_code.model import active_learning as al
from dissertation_code.utils.logging_config import configure_logging
from dissertation_code.utils.seeding import make_reproducible

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    make_reproducible()

    labelled = pipeline.build_labelled_dataset()
    train, test = pipeline.split(labelled)

    # Scenario-based testing (O5).
    model = al.train_static_baseline(train)
    results = scenarios.run_scenarios(model)
    logger.info(
        "=== Scenario testing (match rate %.0f%%) ===",
        100 * scenarios.match_rate(results),
    )
    for r in results:
        logger.info(
            "  [%s] %s -> predicted %s / expected %s",
            "OK" if r.correct else "XX",
            r.scenario.name,
            r.predicted_class,
            r.scenario.expected_class,
        )

    # Active-learning convergence study (O5).
    report = convergence.run_convergence_study(train, test)
    logger.info("=== Convergence study ===")
    logger.info("  full-labelled baseline accuracy: %.3f", report.baseline_accuracy)
    if report.labels_to_target is not None:
        logger.info(
            "  reached target with %d/%d labels (%.0f%% of pool); meets <=%.0f%% budget: %s",
            report.labels_to_target,
            report.total_pool,
            100 * report.fraction_to_target,
            100 * config.LABEL_BUDGET_FRACTION,
            report.meets_target,
        )
    else:
        logger.info(
            "  active learning did not reach the target accuracy within the pool"
        )


if __name__ == "__main__":
    main()
