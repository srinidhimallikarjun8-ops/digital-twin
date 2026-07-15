"""Phase 1 pipeline entry point.

Runs the Sprint 1 foundations end-to-end on the LaSDPC slice:
  load (T+RH only)  ->  unify schema  ->  synthetic comfort labels (PMV + calibrated noise).

Reproducible by construction: the run is seeded via ``make_reproducible`` and all thresholds come
from ``dissertation_code.config``. Diagnostics go through the logger, not ``print``.
"""

from __future__ import annotations

import logging

from dissertation_code import config
from dissertation_code.comfort import synthetic_labels as sl
from dissertation_code.data import schema, sources
from dissertation_code.utils.logging_config import configure_logging
from dissertation_code.utils.seeding import make_reproducible

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    seed = make_reproducible()
    logger.info("Phase 1 pipeline starting (seed=%d)", seed)

    long = sources.load_lasdpc()
    logger.info("Loaded %d T+RH readings (long format)", len(long))

    gridded = schema.resample_long(long)
    logger.info(
        "Resampled to a %s grid: %d readings", config.RESAMPLE_FREQUENCY, len(gridded)
    )

    wide = schema.to_wide(gridded)
    logger.info(
        "Paired into %d co-located T+RH observations across %d zones",
        len(wide),
        wide[schema.ZONE].nunique(),
    )

    labelled = sl.generate_labels(wide)
    comfortable = int(labelled[sl.COMFORT_LABEL].sum())
    agreement = sl.pmv_agreement_rate(labelled)
    logger.info(
        "Generated synthetic labels: %d/%d comfortable (%.1f%%)",
        comfortable,
        len(labelled),
        100.0 * comfortable / len(labelled),
    )
    logger.info(
        "PMV-vote agreement rate: %.3f (expect ~0.33, per Cheung et al. 2019)",
        agreement,
    )
    logger.info("Phase 1 pipeline complete.")


if __name__ == "__main__":
    main()
