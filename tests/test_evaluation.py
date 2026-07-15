"""Tests for the scenario testing and convergence study harnesses."""

import numpy as np
import pandas as pd

from dissertation_code.comfort import synthetic_labels as sl
from dissertation_code.data import schema
from dissertation_code.evaluation import convergence, scenarios
from dissertation_code.model import active_learning as al


def _labelled(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wide = pd.DataFrame(
        {
            schema.TIMESTAMP: pd.date_range("2024-01-16", periods=n, freq="10min"),
            schema.ZONE: 1,
            schema.TEMPERATURE: rng.uniform(15, 31, n),
            schema.RELATIVE_HUMIDITY: rng.uniform(40, 85, n),
        }
    )
    return sl.generate_labels(wide)


def test_scenarios_run_and_score():
    pool = _labelled(400, seed=1)
    model = al.train_static_baseline(pool)
    results = scenarios.run_scenarios(model)
    assert len(results) == len(scenarios.DEFAULT_SCENARIOS)
    rate = scenarios.match_rate(results)
    assert 0.0 <= rate <= 1.0
    # The clearly-hot and clearly-cold scenarios should be classified correctly.
    by_name = {r.scenario.name: r for r in results}
    assert by_name["hot top-floor flat"].correct
    assert by_name["cold dry morning"].correct


def test_convergence_report_structure():
    pool = _labelled(300, seed=1)
    test = _labelled(120, seed=2)
    report = convergence.run_convergence_study(pool, test)
    assert 0.0 <= report.baseline_accuracy <= 1.0
    assert len(report.label_counts) == len(report.accuracies)
    assert report.total_pool == len(pool)
    if report.labels_to_target is not None:
        assert 0.0 < report.fraction_to_target <= 1.0
