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
from dissertation_code.data import bath, schema
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
    data_dir: Path = config.BATH_DATASET_DIR,
    include_external: bool = False,
) -> pd.DataFrame:
    """Load the Bath Connaught Mansions export as a unified, schema-valid long frame (T+RH only).

    Unlike LaSDPC, this source pairs T and RH natively (same row, same timestamp), so callers
    must NOT resample it onto a common grid — see `bath.load_wide`. The long shape here exists
    only to keep one adapter contract across sources.

    Args:
        data_dir: directory holding the quarterly workbooks.
        include_external: keep the two outdoor sensors (excluded by default: nobody occupies
            the outdoors, so they are context for clothing insulation rather than comfort data).
    """
    wide = bath.load_wide(data_dir=data_dir, include_external=include_external)
    return bath.to_long(wide)


def load_bath_wide(
    data_dir: Path = config.BATH_DATASET_DIR,
    include_external: bool = False,
) -> pd.DataFrame:
    """Load Bath directly in wide form, skipping the long round-trip.

    The pipeline uses this for Bath because the export is already paired: melting to long and
    immediately pivoting back would be pure overhead and would drop the native pairing guarantee.
    """
    return bath.load_wide(data_dir=data_dir, include_external=include_external)
