"""Tests for the Fanger PMV implementation against ASHRAE/ISO reference values."""

import pytest

from dissertation_code.comfort import pmv as pmv_model


def test_neutral_conditions_near_zero():
    # ~26 C / 50% RH at sedentary, light clothing sits close to neutral.
    value = pmv_model.pmv(26.0, 50.0)
    assert abs(value) < 0.5


@pytest.mark.parametrize(
    "ta, rh, expected",
    [
        # met=1.2, clo=0.5, v=0.1, tr=ta. The 22C/60% point is the canonical ISO 7730 worked
        # example (PMV = -0.75); the others are cross-checked against the same parameterisation.
        (22.0, 60.0, -0.75),
        (27.0, 60.0, 0.77),
        (23.5, 60.0, -0.30),
    ],
)
def test_known_reference_points(ta, rh, expected):
    value = pmv_model.pmv(ta, rh)
    assert value == pytest.approx(expected, abs=0.1)


def test_monotonic_in_temperature():
    # Hotter air -> higher PMV, all else equal.
    cold = pmv_model.pmv(18.0, 50.0)
    warm = pmv_model.pmv(28.0, 50.0)
    assert cold < warm


def test_comfort_band():
    assert pmv_model.is_comfortable(0.0)
    assert pmv_model.is_comfortable(0.5)
    assert not pmv_model.is_comfortable(0.6)
    assert not pmv_model.is_comfortable(-1.2)


def test_assumptions_affect_result():
    base = pmv_model.pmv(22.0, 50.0)
    warmer_clothing = pmv_model.pmv(
        22.0, 50.0, assumptions=pmv_model.ComfortAssumptions(clothing_insulation=1.0)
    )
    # More insulation at a cool temperature should raise PMV toward neutral/warm.
    assert warmer_clothing > base
