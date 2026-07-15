"""Tests for explanation, narration, recommendation, decision, and scenario evaluation."""

import numpy as np
import pandas as pd
import pytest

from dissertation_code import config
from dissertation_code.comfort import synthetic_labels as sl
from dissertation_code.data import schema
from dissertation_code.explain.narrate import narrate
from dissertation_code.explain.shap_explain import explain_instance
from dissertation_code.model.base import ComfortModel, make_instance
from dissertation_code.recommend import decision as decision_logic
from dissertation_code.recommend import tradeoff
from dissertation_code.recommend.recommender import recommend


@pytest.fixture
def model() -> ComfortModel:
    rng = np.random.default_rng(0)
    n = 400
    wide = pd.DataFrame(
        {
            schema.TIMESTAMP: pd.date_range("2024-01-16", periods=n, freq="10min"),
            schema.ZONE: 1,
            schema.TEMPERATURE: rng.uniform(15, 31, n),
            schema.RELATIVE_HUMIDITY: rng.uniform(40, 85, n),
        }
    )
    labelled = sl.generate_labels(wide)
    return ComfortModel().fit(labelled, labelled[sl.COMFORT_CLASS].to_numpy())


def test_explain_instance_covers_all_features(model):
    attr = explain_instance(model, make_instance(29.0, 70.0))
    assert set(attr.contributions) == set(config.COMFORT_VARS)
    assert attr.predicted_class in config.COMFORT_CLASSES
    # ranked() returns features ordered by absolute contribution.
    assert len(attr.ranked()) == len(config.COMFORT_VARS)


def test_narrate_produces_sentence(model):
    attr = explain_instance(model, make_instance(30.0, 65.0))
    sentence = narrate(attr)
    assert isinstance(sentence, str) and sentence.endswith(".")


def test_recommend_assembles_fields(model):
    rec = recommend(model, make_instance(30.0, 65.0), zone=1, write_audit=False)
    assert rec.predicted_class in config.COMFORT_CLASSES
    assert 0.0 <= rec.confidence <= 1.0
    assert 0.0 <= rec.uncertainty <= 1.0
    assert rec.action
    assert "shap_contributions" in rec.to_audit_payload()


def test_decision_requires_justification(model):
    rec = recommend(model, make_instance(30.0, 65.0), write_audit=False)
    with pytest.raises(ValueError, match="justification is required"):
        decision_logic.record_decision(rec, decision_logic.Decision.CONFIRM, "  ")


def test_decision_writes_audit(model, tmp_path, monkeypatch):
    from dissertation_code.audit import log as audit

    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(config, "AUDIT_LOG_PATH", path)
    rec = recommend(model, make_instance(30.0, 65.0), write_audit=False)
    decision_logic.record_decision(
        rec, decision_logic.Decision.OVERRIDE, "manager preferred to wait"
    )
    events = audit.read_events(path)
    assert events[-1]["human_decision"] == "override"


def test_tradeoff_summary(model):
    recs = [
        recommend(model, make_instance(t, rh), zone=z, write_audit=False)
        for z, (t, rh) in enumerate(
            [(15.0, 50.0), (23.0, 50.0), (30.0, 70.0), (24.0, 55.0)]
        )
    ]
    summary = tradeoff.summarise(recs)
    assert summary.n_zones == 4
    assert summary.n_comfortable + summary.n_need_attention == 4
    assert isinstance(summary.headline(), str)
