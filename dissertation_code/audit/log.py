"""Append-only structured audit log (architecture Layer 7; backend guidelines §9).

Every decision the system makes — recommendation, occupant query, model update, and human
override/defer/confirm — writes one JSON line here. The log is the evidence base for the
dissertation and embodies the conservation principle of explicable, auditable intervention; app
logic never overwrites or deletes it.

The machine explanation (raw SHAP values) is stored here; the plain-language sentence shown to the
user is stored alongside it but never replaces it.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from dissertation_code import config

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """The kinds of auditable events."""

    RECOMMENDATION = "recommendation"
    QUERY = "query"
    MODEL_UPDATE = "model_update"
    HUMAN_DECISION = "human_decision"


def log_event(
    event_type: EventType,
    payload: dict[str, Any],
    path: Path | None = None,
) -> dict[str, Any]:
    """Append one structured, timestamped record to the audit log and return it.

    Args:
        event_type: the category of event (see EventType).
        payload: event-specific fields (zone, inputs, prediction, uncertainty, shap, action,
            human decision + justification, ...).
        path: audit-log file (JSONL); defaults to ``config.AUDIT_LOG_PATH``.

    Returns:
        The full record that was written (with timestamp + event_type injected).
    """
    path = path or config.AUDIT_LOG_PATH
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event_type": event_type.value,
        **payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    logger.debug("audit: wrote %s event", event_type.value)
    return record


def read_events(path: Path | None = None) -> list[dict[str, Any]]:
    """Read all audit records (oldest first). Returns an empty list if the log does not exist."""
    path = path or config.AUDIT_LOG_PATH
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
