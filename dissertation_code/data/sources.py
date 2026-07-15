"""Dataset adapters: map each raw source into the unified long schema.

Architecture Layer 1. Each adapter is a thin, testable function that takes a raw source and
returns a schema-valid long DataFrame. Adding a new source (e.g. the Bath Connaught Mansions
Tinytag export) means writing one adapter here — nothing downstream changes.

The LaSDPC adapter deliberately *reuses* the Sprint-1 EDA loader rather than re-implementing
cleaning, so there is a single definition of "how LaSDPC is loaded and filtered to T+RH".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dissertation_code import config
from dissertation_code.data import schema
from dissertation_code.eda.loader import (
    drop_missing_sensor_values,
    filter_temp_rh,
    load_raw,
)


def load_lasdpc(csv_path: Path = config.LASDPC_DATASET_PATH) -> pd.DataFrame:
    """Load the LaSDPC slice as a unified, schema-valid long frame (T+RH only).

    Reuses the EDA loader for raw parsing, T+RH filtering, and null handling, then renames the
    LaSDPC-specific columns into the canonical schema columns.
    """
    raw = load_raw(csv_path)
    temp_rh = filter_temp_rh(raw)
    clean, _ = drop_missing_sensor_values(temp_rh)

    long = clean.rename(
        columns={
            "date_time": schema.TIMESTAMP,
            "id_enviroment": schema.ZONE,
            "device_type_label": schema.CHANNEL,
            "sensor_value": schema.VALUE,
        }
    )[list(schema.LONG_COLUMNS)]

    return schema.validate_long(long.reset_index(drop=True))


def load_bath(
    *_args, **_kwargs
) -> pd.DataFrame:  # pragma: no cover - data not yet available
    """Adapter for the Bath Connaught Mansions Tinytag export.

    Not implemented: the raw Tinytag files are not yet in the repo (KB open question / risk
    register). The signature is fixed now so the rest of the pipeline can target it; the body
    lands once the export is provided. Placements RHT/ERHT carry both T+RH; T carries temp only.
    """
    raise NotImplementedError(
        "Bath Tinytag export not yet ingested; provide the raw logger files first."
    )
