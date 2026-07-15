"""Tests for the synthetic comfort-label generator (determinism + noise calibration)."""

import numpy as np
import pandas as pd
import pytest

from dissertation_code import config
from dissertation_code.comfort import synthetic_labels as sl
from dissertation_code.data import schema


@pytest.fixture
def wide_fixture() -> pd.DataFrame:
    """Small deterministic T+RH frame spanning cold..hot conditions (no real data)."""
    temps = np.repeat(np.linspace(16.0, 30.0, 15), 20)
    rh = np.tile(np.linspace(45.0, 80.0, 20), 15)
    return pd.DataFrame(
        {
            schema.TIMESTAMP: pd.date_range(
                "2024-01-16", periods=len(temps), freq="5min"
            ),
            schema.ZONE: 1,
            schema.TEMPERATURE: temps,
            schema.RELATIVE_HUMIDITY: rh,
        }
    )


def test_determinism_under_seed(wide_fixture):
    a = sl.generate_labels(wide_fixture)
    b = sl.generate_labels(wide_fixture)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_votes(wide_fixture):
    a = sl.generate_labels(wide_fixture, sl.GeneratorConfig(seed=1))
    b = sl.generate_labels(wide_fixture, sl.GeneratorConfig(seed=2))
    assert not a[sl.SENSATION_VOTE].equals(b[sl.SENSATION_VOTE])


def test_adds_expected_columns(wide_fixture):
    out = sl.generate_labels(wide_fixture)
    for col in (sl.PMV_VALUE, sl.SENSATION_VOTE, sl.COMFORT_LABEL):
        assert col in out.columns
    assert out[sl.SENSATION_VOTE].between(-3, 3).all()
    assert out[sl.COMFORT_LABEL].dtype == bool


def test_zero_noise_recovers_pmv_categories(wide_fixture):
    # With sigma=0 the vote is exactly the rounded PMV, so agreement is perfect.
    out = sl.generate_labels(wide_fixture, sl.GeneratorConfig(noise_sigma=0.0))
    assert sl.pmv_agreement_rate(out) == pytest.approx(1.0)


def test_noise_calibrated_to_cheung_individual_accuracy():
    # Over a broad, balanced condition grid the default sigma should put PMV-vote agreement near
    # Cheung et al. (2019)'s ~1/3 individual-level accuracy (sanity band 0.2-0.5).
    temps = np.repeat(np.linspace(15.0, 31.0, 40), 40)
    rh = np.tile(np.linspace(40.0, 85.0, 40), 40)
    big = pd.DataFrame(
        {
            schema.TIMESTAMP: pd.date_range(
                "2024-01-16", periods=len(temps), freq="1min"
            ),
            schema.ZONE: 1,
            schema.TEMPERATURE: temps,
            schema.RELATIVE_HUMIDITY: rh,
        }
    )
    out = sl.generate_labels(big)
    assert 0.2 <= sl.pmv_agreement_rate(out) <= 0.5


def test_comfort_threshold_from_config(wide_fixture):
    out = sl.generate_labels(wide_fixture)
    expected = out[sl.SENSATION_VOTE].abs() <= config.COMFORT_VOTE_THRESHOLD
    pd.testing.assert_series_equal(out[sl.COMFORT_LABEL], expected, check_names=False)
