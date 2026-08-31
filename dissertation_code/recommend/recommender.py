"""Recommendation assembly (architecture Layer 5).

Ties the model prediction, its uncertainty, the SHAP attribution, and the plain-language sentence
into a single Recommendation, and writes it to the audit log. The dashboard renders this object;
it contains no UI logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from dissertation_code import config
from dissertation_code.audit import log as audit
from dissertation_code.data import schema
from dissertation_code.explain.narrate import narrate
from dissertation_code.explain.shap_explain import Attribution, explain_instance
from dissertation_code.model import active_learning as al
from dissertation_code.model.base import ComfortModel

# Suggested action per predicted comfort class.
_ACTION = {
    config.COMFORT_CLASS_TOO_COOL: (
        f"Consider raising the temperature by ~{config.TEMPERATURE_ADJUSTMENT_STEP:.1f} C."
    ),
    config.COMFORT_CLASS_TOO_WARM: (
        f"Consider lowering the temperature by ~{config.TEMPERATURE_ADJUSTMENT_STEP:.1f} C."
    ),
    config.COMFORT_CLASS_COMFORTABLE: "No change recommended.",
}


@dataclass(frozen=True)
class Recommendation:
    """A single per-zone recommendation with its supporting evidence."""

    zone: Any
    temperature: float
    relative_humidity: float
    predicted_class: str
    confidence: float  # max class probability
    uncertainty: float  # normalised entropy/margin
    triggered: bool  # did a domain trigger fire?
    action: str
    plain_language: str
    attribution: Attribution
    field_meta: dict[str, Any] = field(default_factory=dict)

    def to_audit_payload(self) -> dict[str, Any]:
        """Flatten to a JSON-serialisable audit record (raw SHAP kept here, not shown to user)."""
        return {
            "zone": self.zone,
            "temperature": self.temperature,
            "relative_humidity": self.relative_humidity,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "triggered": self.triggered,
            "recommended_action": self.action,
            "plain_language": self.plain_language,
            "shap_contributions": self.attribution.contributions,
        }


def recommend(
    model: ComfortModel,
    instance: pd.DataFrame,
    zone: Any = None,
    *,
    write_audit: bool = True,
) -> Recommendation:
    """Produce a Recommendation for one (temperature, RH) instance and log it.

    Args:
        model: a fitted ComfortModel.
        instance: a one-row frame with the model features.
        zone: optional zone identifier for the audit record.
        write_audit: whether to append a recommendation event to the audit log.
    """
    proba = model.predict_proba(instance)[0]
    predicted_class = str(model.classes_[proba.argmax()])
    confidence = float(proba.max())
    uncertainty = float(al.uncertainty_score(proba.reshape(1, -1))[0])

    temperature = float(instance.iloc[0][schema.TEMPERATURE])
    relative_humidity = float(instance.iloc[0][schema.RELATIVE_HUMIDITY])
    triggered = al.pmv_trigger(
        temperature, relative_humidity
    ) or al.instantaneous_humidity_trigger(relative_humidity)

    attribution = explain_instance(model, instance)
    plain_language = narrate(attribution)

    rec = Recommendation(
        zone=zone,
        temperature=temperature,
        relative_humidity=relative_humidity,
        predicted_class=predicted_class,
        confidence=confidence,
        uncertainty=uncertainty,
        triggered=triggered,
        action=_ACTION.get(predicted_class, "No change recommended."),
        plain_language=plain_language,
        attribution=attribution,
    )
    if write_audit:
        audit.log_event(audit.EventType.RECOMMENDATION, rec.to_audit_payload())
    return rec
