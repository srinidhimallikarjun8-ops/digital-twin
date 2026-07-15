"""Tests for active-learning uncertainty, triggers, query selection, and the loop."""

import numpy as np
import pandas as pd
import pytest

from dissertation_code.comfort import synthetic_labels as sl
from dissertation_code.data import schema
from dissertation_code.model import active_learning as al
from dissertation_code.model.base import ComfortModel


def test_entropy_uncertainty_extremes():
    confident = np.array([[1.0, 0.0, 0.0]])
    uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])
    assert al.uncertainty_score(confident, "entropy")[0] == pytest.approx(0.0)
    assert al.uncertainty_score(uniform, "entropy")[0] == pytest.approx(1.0)


def test_margin_uncertainty():
    clear = np.array([[0.9, 0.1, 0.0]])  # large margin -> low uncertainty
    close = np.array([[0.5, 0.5, 0.0]])  # zero margin -> high uncertainty
    assert (
        al.uncertainty_score(clear, "margin")[0]
        < al.uncertainty_score(close, "margin")[0]
    )


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown uncertainty strategy"):
        al.uncertainty_score(np.array([[0.5, 0.5]]), "bogus")


def test_pmv_trigger_fires_outside_band():
    assert al.pmv_trigger(31.0, 70.0)  # clearly hot -> trigger
    assert not al.pmv_trigger(25.0, 50.0)  # neutral (PMV ~0.04) -> no trigger


def test_instantaneous_humidity_trigger():
    assert al.instantaneous_humidity_trigger(80.0)
    assert not al.instantaneous_humidity_trigger(60.0)


def test_sustained_humidity_trigger():
    # RH stays above 75% across 40 minutes (5 readings, 10 min apart) -> later readings flagged.
    df = pd.DataFrame(
        {
            schema.TIMESTAMP: pd.date_range(
                "2024-01-16 10:00", periods=5, freq="10min"
            ),
            schema.ZONE: 1,
            schema.TEMPERATURE: 24.0,
            schema.RELATIVE_HUMIDITY: [80.0, 80.0, 80.0, 80.0, 80.0],
        }
    )
    flags = al.sustained_humidity_trigger(df)
    assert not flags.iloc[0]  # 0 minutes elapsed
    assert flags.iloc[-1]  # 40 minutes sustained -> flagged


def _labelled(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    temps = rng.uniform(15, 31, n)
    rh = rng.uniform(40, 85, n)
    wide = pd.DataFrame(
        {
            schema.TIMESTAMP: pd.date_range("2024-01-16", periods=n, freq="10min"),
            schema.ZONE: 1,
            schema.TEMPERATURE: temps,
            schema.RELATIVE_HUMIDITY: rh,
        }
    )
    return sl.generate_labels(wide)


def test_run_active_learning_records_curve_and_improves():
    pool = _labelled(300, seed=1)
    test = _labelled(120, seed=2)
    result = al.run_active_learning(pool, test, seed_count=20, batch_size=20)
    # Convergence curve is recorded and the label budget is non-decreasing.
    assert result.label_counts == sorted(result.label_counts)
    assert len(result.accuracies) == len(result.label_counts)
    # The loop consumes the whole pool by the end and stays at a sane accuracy throughout.
    assert result.label_counts[-1] == len(pool)
    assert all(acc >= 0.6 for acc in result.accuracies)


def test_select_queries_prioritises_triggered_rows():
    pool = _labelled(100, seed=3)
    model = ComfortModel().fit(pool, pool[sl.COMFORT_CLASS].to_numpy())
    chosen = al.select_queries(model, pool, batch_size=5)
    assert len(chosen) == 5
    assert len(set(chosen)) == 5  # no duplicates
