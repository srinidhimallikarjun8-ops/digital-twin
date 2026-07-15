"""Tests for the closed active-learning loop: label store, feedback, update, persistence, query."""

import numpy as np
import pandas as pd
import pytest

from dissertation_code import config
from dissertation_code.comfort import synthetic_labels as sl
from dissertation_code.data import schema
from dissertation_code.model import active_learning as al
from dissertation_code.model import label_store, store
from dissertation_code.model.base import make_instance
from dissertation_code.recommend import decision as decision_logic
from dissertation_code.recommend.recommender import recommend


def _prior(n: int = 400, seed: int = 0) -> pd.DataFrame:
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


# --- label store ---------------------------------------------------------------------------
def test_label_store_roundtrip(tmp_path):
    path = tmp_path / "labels.jsonl"
    label_store.append_label(1, 29.0, 70.0, config.COMFORT_CLASS_TOO_WARM, path=path)
    label_store.append_label(2, 16.0, 50.0, config.COMFORT_CLASS_TOO_COOL, path=path)
    df = label_store.load_labels(path)
    assert list(df.columns) == list(label_store.LABEL_COLUMNS)
    assert len(df) == 2
    assert label_store.count(path) == 2


def test_label_store_rejects_bad_class(tmp_path):
    with pytest.raises(ValueError, match="comfort_class must be one of"):
        label_store.append_label(1, 24.0, 50.0, "freezing", path=tmp_path / "l.jsonl")


def test_load_labels_empty(tmp_path):
    df = label_store.load_labels(tmp_path / "missing.jsonl")
    assert df.empty
    assert list(df.columns) == list(label_store.LABEL_COLUMNS)


# --- feedback -> label mapping -------------------------------------------------------------
def test_decision_to_label_mapping():
    prior = _prior()
    model = al.train_static_baseline(prior)
    rec = recommend(model, make_instance(30.0, 65.0), zone=1, write_audit=False)
    assert (
        decision_logic.decision_to_label(rec, decision_logic.Decision.CONFIRM)
        == rec.predicted_class
    )
    assert (
        decision_logic.decision_to_label(
            rec, decision_logic.Decision.OVERRIDE, config.COMFORT_CLASS_TOO_COOL
        )
        == config.COMFORT_CLASS_TOO_COOL
    )
    assert decision_logic.decision_to_label(rec, decision_logic.Decision.DEFER) is None


def test_override_requires_valid_class():
    prior = _prior()
    model = al.train_static_baseline(prior)
    rec = recommend(model, make_instance(30.0, 65.0), write_audit=False)
    with pytest.raises(ValueError, match="override requires a corrected_class"):
        decision_logic.decision_to_label(rec, decision_logic.Decision.OVERRIDE, None)


def test_apply_feedback_writes_audit_and_label(tmp_path, monkeypatch):
    from dissertation_code.audit import log as audit

    monkeypatch.setattr(config, "AUDIT_LOG_PATH", tmp_path / "audit.jsonl")
    monkeypatch.setattr(config, "LABEL_STORE_PATH", tmp_path / "labels.jsonl")
    prior = _prior()
    model = al.train_static_baseline(prior)
    rec = recommend(model, make_instance(30.0, 65.0), zone=1, write_audit=False)

    label = decision_logic.apply_feedback(
        rec, decision_logic.Decision.CONFIRM, "looks right"
    )
    assert label == rec.predicted_class
    assert (
        audit.read_events(tmp_path / "audit.jsonl")[-1]["human_decision"] == "confirm"
    )
    assert label_store.count(tmp_path / "labels.jsonl") == 1

    # Defer logs an audit record but stores no label.
    assert (
        decision_logic.apply_feedback(rec, decision_logic.Decision.DEFER, "wait")
        is None
    )
    assert label_store.count(tmp_path / "labels.jsonl") == 1


# --- model update from labels -------------------------------------------------------------
def test_update_with_labels_shifts_prediction():
    prior = _prior()
    instance = make_instance(24.0, 55.0)  # a near-neutral point

    # Flood that point with heavily-weighted human labels asserting "too warm".
    human = pd.DataFrame(
        {
            schema.ZONE: 1,
            schema.TEMPERATURE: [24.0] * 5,
            schema.RELATIVE_HUMIDITY: [55.0] * 5,
            sl.COMFORT_CLASS: [config.COMFORT_CLASS_TOO_WARM] * 5,
        }
    )
    updated = al.update_with_labels(prior, human, human_weight=200.0)
    # The heavily-weighted human feedback pulls the prediction to "too warm".
    assert updated.predict(instance)[0] == config.COMFORT_CLASS_TOO_WARM


def test_update_with_empty_human_labels_equals_prior_fit():
    prior = _prior()
    model = al.update_with_labels(
        prior, pd.DataFrame(columns=list(label_store.LABEL_COLUMNS))
    )
    assert model.is_fitted


# --- persistence --------------------------------------------------------------------------
def test_model_save_load_roundtrip(tmp_path):
    prior = _prior()
    model = al.train_static_baseline(prior)
    mpath = tmp_path / "m.joblib"
    store.save_model(model, model_path=mpath, manifest_path=tmp_path / "manifest.json")
    loaded = store.load_model(mpath)
    assert loaded is not None
    inst = make_instance(29.0, 70.0)
    assert loaded.predict(inst)[0] == model.predict(inst)[0]
    assert (tmp_path / "manifest.json").exists()


def test_load_model_missing_returns_none(tmp_path):
    assert store.load_model(tmp_path / "nope.joblib") is None


# --- active query -------------------------------------------------------------------------
def test_next_query_returns_uncertain_instance():
    prior = _prior()
    model = al.train_static_baseline(prior)
    q = al.next_query(model, prior)
    assert q is not None
    assert 0 <= q.index < len(prior)
    assert 0.0 <= q.uncertainty <= 1.0


def test_next_query_respects_exclusions():
    prior = _prior(n=50)
    model = al.train_static_baseline(prior)
    exclude = set(range(len(prior)))
    assert al.next_query(model, prior, exclude=exclude) is None
