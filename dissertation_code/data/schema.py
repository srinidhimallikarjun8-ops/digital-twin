"""Canonical schema for the unified temperature + relative-humidity time series.

Architecture Layer 1 (data & ingestion). Every data source (LaSDPC IoT, Bath Connaught
Mansions) is adapted into a single *long* DataFrame with the columns defined here, so that
everything downstream (labelling, active learning, evaluation) is source-agnostic.

Hard scope constraint: only temperature + relative humidity are represented. CO2 / light are
deliberately absent from this schema — they are interface-only context, never model data.
"""

from __future__ import annotations

import pandas as pd

from dissertation_code import config

# --- Column names (single source of truth; import these, never hard-code strings) -----------
TIMESTAMP = "timestamp"
ZONE = "zone"
TEMPERATURE, RELATIVE_HUMIDITY = config.COMFORT_VARS  # degrees Celsius, percent (0-100)

#: A reading in the unified long format is one (timestamp, zone, channel, value) row.
CHANNEL = "channel"
VALUE = "value"

#: The only two channels permitted in the modelling pipeline.
TEMPERATURE_CHANNEL = TEMPERATURE
HUMIDITY_CHANNEL = RELATIVE_HUMIDITY
MODEL_CHANNELS = (TEMPERATURE_CHANNEL, HUMIDITY_CHANNEL)

LONG_COLUMNS = (TIMESTAMP, ZONE, CHANNEL, VALUE)
WIDE_COLUMNS = (TIMESTAMP, ZONE, TEMPERATURE, RELATIVE_HUMIDITY)

# Plausible physical bounds (from config), used only for validation (not silent clipping).
TEMPERATURE_BOUNDS = config.PHYSICAL_TEMPERATURE_BOUNDS
RELATIVE_HUMIDITY_BOUNDS = config.PHYSICAL_HUMIDITY_BOUNDS


class SchemaError(ValueError):
    """Raised when a DataFrame does not conform to the unified schema."""


def validate_long(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a long-format reading frame; return it unchanged if valid.

    Checks columns, channel membership, and physical bounds. Raises SchemaError on violation
    rather than coercing, so data-quality problems surface loudly (risk-register: dataset quality).
    """
    missing = [c for c in LONG_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"long frame missing columns: {missing}")

    bad_channels = set(df[CHANNEL].unique()) - set(MODEL_CHANNELS)
    if bad_channels:
        raise SchemaError(
            f"long frame contains non-model channels {bad_channels}; "
            f"only {MODEL_CHANNELS} are permitted (T+RH-only scope)"
        )

    _check_bounds(df, TEMPERATURE_CHANNEL, TEMPERATURE_BOUNDS)
    _check_bounds(df, HUMIDITY_CHANNEL, RELATIVE_HUMIDITY_BOUNDS)
    return df


def _check_bounds(df: pd.DataFrame, channel: str, bounds: tuple[float, float]) -> None:
    lo, hi = bounds
    values = df.loc[df[CHANNEL] == channel, VALUE].dropna()
    out_of_range = values[(values < lo) | (values > hi)]
    if not out_of_range.empty:
        raise SchemaError(
            f"{channel}: {len(out_of_range)} value(s) outside physical bounds {bounds} "
            f"(e.g. {out_of_range.iloc[0]})"
        )


def resample_long(
    df_long: pd.DataFrame, freq: str = config.RESAMPLE_FREQUENCY
) -> pd.DataFrame:
    """Resample each (zone, channel) onto a common regular time grid by mean.

    Raw temperature and humidity arrive from separate devices at unaligned sub-minute timestamps,
    so an exact-timestamp pivot pairs almost nothing. Resampling both channels onto the same grid
    (default ``config.RESAMPLE_FREQUENCY``) lets a later ``to_wide`` pair co-located T+RH cleanly.
    Empty grid buckets are dropped (no interpolation across gaps at this stage).

    Args:
        df_long: a schema-valid long frame.
        freq: a pandas offset alias for the target grid (e.g. "10min").

    Returns:
        A schema-valid long frame with timestamps snapped to the grid.
    """
    validate_long(df_long)
    df = df_long.copy()
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
    resampled = (
        df.set_index(TIMESTAMP)
        .groupby([ZONE, CHANNEL])[VALUE]
        .resample(freq)
        .mean()
        .reset_index()
        .dropna(subset=[VALUE])
    )
    return validate_long(resampled[list(LONG_COLUMNS)].reset_index(drop=True))


def to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot a validated long frame to one row per (timestamp, zone) with T and RH columns.

    This is the shape the modelling layer consumes: each row is a co-located T+RH observation.
    Rows where either channel is missing are dropped, since a comfort prediction needs both.
    """
    validate_long(df_long)
    wide = (
        df_long.pivot_table(
            index=[TIMESTAMP, ZONE], columns=CHANNEL, values=VALUE, aggfunc="mean"
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    for col in (TEMPERATURE, RELATIVE_HUMIDITY):
        if col not in wide.columns:
            wide[col] = pd.NA
    wide = wide.dropna(subset=[TEMPERATURE, RELATIVE_HUMIDITY])
    return (
        wide[list(WIDE_COLUMNS)].sort_values([ZONE, TIMESTAMP]).reset_index(drop=True)
    )
