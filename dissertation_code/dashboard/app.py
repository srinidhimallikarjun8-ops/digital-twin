"""Streamlit closed-loop decision-support dashboard (architecture Layer 6).

Run with: ``uv run streamlit run dissertation_code/dashboard/app.py``

This is the live human-in-the-loop active-learning interface. It has no business logic (backend
guidelines §4): it calls into the model / explanation / recommendation / pipeline packages, renders
results, and closes the loop — every confirm/override/query answer becomes a stored label and the
model is retrained so the *next* prediction reflects it.
"""

# ruff: noqa: E402  (imports below intentionally follow the sys.path bootstrap)
from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` puts this script's own directory on sys.path, not the repo root, so add the
# repo root to make the `dissertation_code` package importable regardless of how the app is launched.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dissertation_code import config, pipeline
from dissertation_code.audit import log as audit
from dissertation_code.model import active_learning as al
from dissertation_code.model import label_store
from dissertation_code.model.base import make_instance
from dissertation_code.recommend import decision as decision_logic
from dissertation_code.recommend import tradeoff
from dissertation_code.recommend.recommender import recommend
from dissertation_code.utils.seeding import make_reproducible

_CLASS_ICON = {
    config.COMFORT_CLASS_COMFORTABLE: "🟢",
    config.COMFORT_CLASS_TOO_WARM: "🔴",
    config.COMFORT_CLASS_TOO_COOL: "🔵",
}
_CLASS_LABELS = [c.replace("_", " ") for c in config.COMFORT_CLASSES]


@st.cache_data
def _synthetic_prior() -> pd.DataFrame:
    """The PMV-derived synthetic pool (loaded once; expensive)."""
    make_reproducible()
    return pipeline.build_labelled_dataset()


def _get_model(prior: pd.DataFrame):
    if "model" not in st.session_state:
        st.session_state.model = pipeline.load_or_train_live_model(prior)
    return st.session_state.model


def _retrain(prior: pd.DataFrame) -> None:
    st.session_state.model = pipeline.retrain_live_model(prior)


def _class_from_label(label: str) -> str:
    return label.replace(" ", "_")


def _render_recommendation(rec) -> None:
    icon = _CLASS_ICON.get(rec.predicted_class, "⚪")
    c1, c2 = st.columns(2)
    c1.metric("Predicted comfort", f"{icon} {rec.predicted_class.replace('_', ' ')}")
    c2.metric("Confidence", f"{rec.confidence:.0%}")
    st.progress(
        min(rec.uncertainty, 1.0), text=f"Model uncertainty: {rec.uncertainty:.0%}"
    )
    if rec.triggered:
        st.warning(
            "A domain trigger fired (outside comfort band or high humidity) — review advised."
        )
    st.info(f"**Action:** {rec.action}")
    st.write(f"**Why:** {rec.plain_language}")
    with st.expander("Supporting evidence (SHAP — audit detail)"):
        st.dataframe(
            pd.DataFrame(
                {
                    "feature": list(rec.attribution.contributions),
                    "value": [
                        rec.attribution.feature_values[f]
                        for f in rec.attribution.contributions
                    ],
                    "shap_contribution": list(rec.attribution.contributions.values()),
                }
            ),
            hide_index=True,
        )


def _check_tab(prior: pd.DataFrame, model) -> None:
    st.sidebar.header("Conditions")
    zone = st.sidebar.selectbox("Zone", sorted(prior["zone"].unique()))
    temperature = st.sidebar.slider("Temperature (°C)", 10.0, 35.0, 24.0, 0.1)
    relative_humidity = st.sidebar.slider(
        "Relative humidity (%)", 20.0, 100.0, 55.0, 1.0
    )

    rec = recommend(
        model,
        make_instance(temperature, relative_humidity),
        zone=zone,
        write_audit=False,
    )
    st.subheader("Recommendation")
    _render_recommendation(rec)

    st.subheader("Your decision (this updates the model)")
    choice = st.radio(
        "Action", [d.value for d in decision_logic.Decision], horizontal=True
    )
    corrected = None
    if choice == decision_logic.Decision.OVERRIDE.value:
        corrected_label = st.selectbox("Actual comfort", _CLASS_LABELS)
        corrected = _class_from_label(corrected_label)
    justification = st.text_input("Justification (required)")

    if st.button("Record decision & update model"):
        if not justification.strip():
            st.error("A justification is required for every decision (auditability).")
        else:
            audit.log_event(audit.EventType.RECOMMENDATION, rec.to_audit_payload())
            label = decision_logic.apply_feedback(
                rec,
                decision_logic.Decision(choice),
                justification,
                corrected_class=corrected,
            )
            if label is not None:
                _retrain(prior)
                st.success(
                    f"Recorded and model retrained — new label: {label.replace('_', ' ')}."
                )
            else:
                st.info("Decision deferred — logged, no label stored, model unchanged.")


def _query_tab(prior: pd.DataFrame, model) -> None:
    st.subheader("Active query — the system asks what it is least sure about")
    asked = st.session_state.setdefault("asked", set())
    query = al.next_query(model, prior, exclude=asked)

    if query is None:
        st.success("Nothing left to ask in the pool.")
        return
    if query.uncertainty < config.UNCERTAINTY_STOP_THRESHOLD:
        st.success(
            f"Most-uncertain remaining case is below the stop threshold "
            f"({query.uncertainty:.0%} < {config.UNCERTAINTY_STOP_THRESHOLD:.0%}) — the model has converged."
        )

    st.write(
        f"**Zone {query.zone}** — temperature **{query.temperature:.1f} °C**, "
        f"humidity **{query.relative_humidity:.0f}%** (model uncertainty {query.uncertainty:.0%}, "
        f"current guess: {query.predicted_class.replace('_', ' ')})"
    )
    answer_label = st.selectbox(
        "How would this feel?", _CLASS_LABELS, key="query_answer"
    )
    if st.button("Submit answer"):
        label_store.append_label(
            zone=query.zone,
            temperature=query.temperature,
            relative_humidity=query.relative_humidity,
            comfort_class=_class_from_label(answer_label),
            source="human_query",
        )
        asked.add(query.index)
        _retrain(prior)
        st.success("Answer stored and model retrained — next query will reflect it.")
        st.rerun()


def _zone_overview(prior: pd.DataFrame, model) -> None:
    """Multi-zone trade-off summary using the latest reading per zone."""
    latest = prior.sort_values("timestamp").groupby("zone").tail(1)
    recs = [
        recommend(
            model,
            make_instance(r["temperature"], r["relative_humidity"]),
            zone=r["zone"],
            write_audit=False,
        )
        for _, r in latest.iterrows()
    ]
    summary = tradeoff.summarise(recs)
    st.caption(f"Multi-zone status (latest readings): {summary.headline()}")


def main() -> None:
    st.set_page_config(page_title="Heritage Comfort Decision Support", page_icon="🏛️")
    st.title("🏛️ Heritage Comfort Decision Support")
    st.caption(
        "Proof of concept — temperature + humidity only; comfort labels are synthetic "
        "(validate the mechanics, not real-occupant accuracy)."
    )

    prior = _synthetic_prior()
    model = _get_model(prior)

    _zone_overview(prior, model)
    st.sidebar.metric("Human labels collected", label_store.count())

    check, query = st.tabs(["Check a condition", "Answer a query (active learning)"])
    with check:
        _check_tab(prior, model)
    with query:
        _query_tab(prior, model)

    with st.expander("Recent audit log"):
        events = audit.read_events()[-10:]
        st.dataframe(pd.DataFrame(events), hide_index=True) if events else st.write(
            "No records yet."
        )


main()
