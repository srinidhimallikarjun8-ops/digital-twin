"""Human override / confirm / defer logic (architecture Layer 5; backend guidelines §9).

Every recommendation is subject to a logged human decision — the core of the heritage
"explicable, reversible intervention" requirement. Each decision writes one audit record carrying
the justification, so the trail shows not just what the system suggested but what the human chose
and why.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from dissertation_code import config
from dissertation_code.audit import log as audit
from dissertation_code.model import label_store
from dissertation_code.recommend.recommender import Recommendation


class Decision(str, Enum):
    """The three permitted human responses to a recommendation."""

    CONFIRM = "confirm"  # accept and act on the recommendation
    OVERRIDE = "override"  # reject / do something different
    DEFER = "defer"  # postpone the decision


def decision_to_label(
    recommendation: Recommendation,
    decision: Decision,
    corrected_class: str | None = None,
) -> str | None:
    """Map a human decision to a comfort-class training label (or None if no label results).

    - confirm  -> the predicted class is endorsed and becomes the label.
    - override -> the human-supplied corrected class becomes the label (required).
    - defer    -> no label.
    """
    if decision is Decision.CONFIRM:
        return recommendation.predicted_class
    if decision is Decision.OVERRIDE:
        if corrected_class not in config.COMFORT_CLASSES:
            raise ValueError(
                f"override requires a corrected_class in {config.COMFORT_CLASSES}; got {corrected_class!r}"
            )
        return corrected_class
    return None  # defer


def record_decision(
    recommendation: Recommendation,
    decision: Decision,
    justification: str,
) -> dict[str, Any]:
    """Append a human-decision record to the audit log and return it.

    Args:
        recommendation: the recommendation being acted on.
        decision: confirm / override / defer.
        justification: free-text reason (required — an empty reason is rejected).

    Raises:
        ValueError: if no justification is given (auditability requirement).
    """
    if not justification or not justification.strip():
        raise ValueError("a justification is required for every human decision")

    payload = {
        "zone": recommendation.zone,
        "temperature": recommendation.temperature,
        "relative_humidity": recommendation.relative_humidity,
        "recommended_action": recommendation.action,
        "predicted_class": recommendation.predicted_class,
        "human_decision": decision.value,
        "justification": justification.strip(),
    }
    return audit.log_event(audit.EventType.HUMAN_DECISION, payload)


def apply_feedback(
    recommendation: Recommendation,
    decision: Decision,
    justification: str,
    corrected_class: str | None = None,
) -> str | None:
    """Record the decision to the audit log AND store the resulting training label.

    This is the bridge that closes the active-learning loop: it logs the human decision (audit
    trail) and, when the decision yields a label (confirm/override), appends it to the label store
    so the next retrain incorporates it.

    Returns:
        The stored comfort-class label, or None for a defer.
    """
    record_decision(recommendation, decision, justification)
    label = decision_to_label(recommendation, decision, corrected_class)
    if label is not None:
        label_store.append_label(
            zone=recommendation.zone,
            temperature=recommendation.temperature,
            relative_humidity=recommendation.relative_humidity,
            comfort_class=label,
            source=f"human_{decision.value}",
        )
    return label
