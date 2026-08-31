"""Append-only store of human comfort labels (active-learning item 1).

When an occupant answers a query or overrides a recommendation, the resulting (zone, T, RH,
comfort_class) becomes a reusable *training label* — distinct from the audit log, which records the
human-readable decision. These labels are what personalise the model away from the PMV prior.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from dissertation_code import config
from dissertation_code.comfort import synthetic_labels as sl
from dissertation_code.data import schema

#: Columns of the DataFrame returned by load_labels (the features + the class target).
LABEL_COLUMNS = (
    schema.ZONE,
    schema.TEMPERATURE,
    schema.RELATIVE_HUMIDITY,
    sl.COMFORT_CLASS,
)


def append_label(
    zone: Any,
    temperature: float,
    relative_humidity: float,
    comfort_class: str,
    source: str = "human",
    path: Path | None = None,
) -> dict[str, Any]:
    """Append one labelled observation to the store and return the written record.

    Args:
        zone: zone identifier.
        temperature: deg C.
        relative_humidity: percent.
        comfort_class: one of config.COMFORT_CLASSES.
        source: provenance tag (e.g. "human", "human_override", "human_query").
        path: store file; defaults to config.LABEL_STORE_PATH.
    """
    if comfort_class not in config.COMFORT_CLASSES:
        raise ValueError(
            f"comfort_class must be one of {config.COMFORT_CLASSES}; got {comfort_class!r}"
        )

    path = path or config.LABEL_STORE_PATH
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        schema.ZONE: zone,
        schema.TEMPERATURE: float(temperature),
        schema.RELATIVE_HUMIDITY: float(relative_humidity),
        sl.COMFORT_CLASS: comfort_class,
        "source": source,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record


def load_labels(path: Path | None = None) -> pd.DataFrame:
    """Load all human labels as a DataFrame with LABEL_COLUMNS (empty frame if none yet)."""
    path = path or config.LABEL_STORE_PATH
    if not path.exists():
        return pd.DataFrame(columns=list(LABEL_COLUMNS))
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        return pd.DataFrame(columns=list(LABEL_COLUMNS))
    return pd.DataFrame(rows)[list(LABEL_COLUMNS)]


def count(path: Path | None = None) -> int:
    """Number of stored human labels."""
    return len(load_labels(path))
