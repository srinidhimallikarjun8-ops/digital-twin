"""Tests for the append-only audit log."""

from dissertation_code.audit import log as audit


def test_append_and_read(tmp_path):
    path = tmp_path / "audit.jsonl"
    audit.log_event(
        audit.EventType.RECOMMENDATION, {"zone": 1, "action": "cool"}, path=path
    )
    audit.log_event(
        audit.EventType.HUMAN_DECISION,
        {"zone": 1, "human_decision": "confirm"},
        path=path,
    )

    events = audit.read_events(path)
    assert len(events) == 2
    assert events[0]["event_type"] == "recommendation"
    assert events[1]["human_decision"] == "confirm"
    assert "timestamp" in events[0]


def test_append_only(tmp_path):
    path = tmp_path / "audit.jsonl"
    audit.log_event(audit.EventType.QUERY, {"a": 1}, path=path)
    audit.log_event(audit.EventType.QUERY, {"a": 2}, path=path)
    # Second write must not overwrite the first.
    assert [e["a"] for e in audit.read_events(path)] == [1, 2]


def test_read_missing_returns_empty(tmp_path):
    assert audit.read_events(tmp_path / "nope.jsonl") == []
