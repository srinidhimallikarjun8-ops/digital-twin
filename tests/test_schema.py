"""Tests for the unified data schema validation and long->wide pivot."""

import pandas as pd
import pytest

from dissertation_code.data import schema


def _long(rows):
    return pd.DataFrame(rows, columns=list(schema.LONG_COLUMNS))


def test_valid_long_passes():
    df = _long(
        [
            ("2024-01-16 10:00", 1, schema.TEMPERATURE_CHANNEL, 22.0),
            ("2024-01-16 10:00", 1, schema.HUMIDITY_CHANNEL, 55.0),
        ]
    )
    assert schema.validate_long(df) is df


def test_rejects_non_model_channel():
    df = _long([("2024-01-16 10:00", 1, "energy_co2_proxy", 800.0)])
    with pytest.raises(schema.SchemaError, match="non-model channels"):
        schema.validate_long(df)


def test_rejects_out_of_bounds_humidity():
    df = _long([("2024-01-16 10:00", 1, schema.HUMIDITY_CHANNEL, 140.0)])
    with pytest.raises(schema.SchemaError, match="physical bounds"):
        schema.validate_long(df)


def test_resample_aligns_unaligned_channels():
    # T and RH arrive a few seconds apart (separate devices); resampling onto a 10-min grid
    # snaps them into the same bucket so they can be paired.
    df = _long(
        [
            ("2024-01-16 10:00:03", 1, schema.TEMPERATURE_CHANNEL, 22.0),
            ("2024-01-16 10:00:07", 1, schema.HUMIDITY_CHANNEL, 55.0),
            ("2024-01-16 10:09:50", 1, schema.TEMPERATURE_CHANNEL, 24.0),
            ("2024-01-16 10:09:55", 1, schema.HUMIDITY_CHANNEL, 57.0),
        ]
    )
    assert schema.to_wide(df).empty  # exact pivot pairs nothing
    wide = schema.to_wide(schema.resample_long(df, freq="10min"))
    assert (
        len(wide) == 1
    )  # both readings fall in the 10:00 bucket -> one paired observation
    assert wide.iloc[0][schema.TEMPERATURE] == pytest.approx(23.0)  # mean of 22 and 24
    assert wide.iloc[0][schema.RELATIVE_HUMIDITY] == pytest.approx(56.0)


def test_to_wide_pairs_temp_and_rh():
    df = _long(
        [
            ("2024-01-16 10:00", 1, schema.TEMPERATURE_CHANNEL, 22.0),
            ("2024-01-16 10:00", 1, schema.HUMIDITY_CHANNEL, 55.0),
            ("2024-01-16 10:05", 1, schema.TEMPERATURE_CHANNEL, 23.0),  # no RH partner
        ]
    )
    wide = schema.to_wide(df)
    assert list(wide.columns) == list(schema.WIDE_COLUMNS)
    assert len(wide) == 1  # only the fully-paired observation survives
    assert wide.iloc[0][schema.TEMPERATURE] == 22.0
    assert wide.iloc[0][schema.RELATIVE_HUMIDITY] == 55.0
