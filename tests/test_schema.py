from __future__ import annotations

import pytest

from or_pr_review.errors import LaneError, SchemaError
from or_pr_review.schema import (
    parse_finding,
    parse_lane_artifact,
    parse_model_findings,
)


def test_parse_finding_happy_path() -> None:
    finding = parse_finding(
        {
            "title": "Missing auth check",
            "body": "The handler accepts unauthenticated POSTs.",
            "severity": "bug",
            "file": "src/api.py",
            "line": 42,
        },
        "x-ai/grok-4.6",
    )
    assert finding.title == "Missing auth check"
    assert finding.severity == "bug"
    assert finding.file == "src/api.py"
    assert finding.line == 42
    assert finding.model_id == "x-ai/grok-4.6"


def test_parse_finding_accepts_path_alias_and_null_location() -> None:
    finding = parse_finding(
        {
            "title": "Docs typo",
            "body": "README says 'teh'.",
            "severity": "NIT",
            "path": None,
            "line": None,
        },
        "anthropic/claude-sonnet-4.6",
    )
    assert finding.severity == "nit"
    assert finding.file is None
    assert finding.line is None


def test_parse_model_findings_from_fenced_json() -> None:
    text = """Here you go
```json
{"findings": [{"title": "Race", "body": "Check-then-act", "severity": "risk", "file": "db.py", "line": 9}]}
```
"""
    findings = parse_model_findings(text, "x-ai/grok-4.6")
    assert len(findings) == 1
    assert findings[0].title == "Race"


def test_parse_model_findings_empty_array() -> None:
    assert parse_model_findings({"findings": []}, "x-ai/grok-4.6") == []


def test_parse_model_findings_missing_array_is_lane_error() -> None:
    with pytest.raises(LaneError, match="missing a findings array"):
        parse_model_findings({"issues": []}, "x-ai/grok-4.6")


def test_parse_model_findings_bad_severity_is_lane_error() -> None:
    with pytest.raises(LaneError, match="severity"):
        parse_model_findings(
            {
                "findings": [
                    {
                        "title": "x",
                        "body": "y",
                        "severity": "critical",
                        "file": None,
                        "line": None,
                    }
                ]
            },
            "x-ai/grok-4.6",
        )


def test_lane_artifact_schema_mismatch_fail_closed() -> None:
    with pytest.raises(SchemaError, match="schema_version"):
        parse_lane_artifact({"ok": True, "model": "x-ai/grok-4.6", "findings": [], "error": None})
    with pytest.raises(SchemaError, match="missing required keys"):
        parse_lane_artifact({"schema_version": 1, "ok": True, "model": "x-ai/grok-4.6"})
    with pytest.raises(SchemaError, match="ok must be a boolean"):
        parse_lane_artifact(
            {
                "schema_version": 1,
                "ok": "yes",
                "model": "x-ai/grok-4.6",
                "findings": [],
                "error": None,
            }
        )


def test_lane_artifact_invalid_finding_fail_closed() -> None:
    with pytest.raises(SchemaError, match="finding is invalid"):
        parse_lane_artifact(
            {
                "schema_version": 1,
                "ok": True,
                "model": "x-ai/grok-4.6",
                "findings": [{"title": "only-title"}],
                "error": None,
            }
        )


def test_lane_artifact_happy_path() -> None:
    result = parse_lane_artifact(
        {
            "schema_version": 1,
            "ok": True,
            "model": "x-ai/grok-4.6",
            "findings": [
                {
                    "title": "Leak",
                    "body": "Pointer not freed",
                    "severity": "bug",
                    "file": "mem.c",
                    "line": 3,
                    "model_id": "x-ai/grok-4.6",
                }
            ],
            "error": None,
            "elapsed_ms": 1200,
        }
    )
    assert result.ok
    assert result.findings[0].title == "Leak"
    assert result.elapsed_ms == 1200


def test_unsafe_finding_paths_are_dropped_not_fatal() -> None:
    from or_pr_review.schema import parse_finding

    for bad in ("../etc/passwd", "a`b.py", "/abs/path.py", "dir\\file.py", "a\x00b"):
        finding = parse_finding(
            {"title": "t", "body": "b", "severity": "bug", "file": bad, "line": 3},
            "x-ai/grok-4.6",
        )
        assert finding.file is None, bad
        assert finding.title == "t"
    ok = parse_finding(
        {"title": "t", "body": "b", "severity": "bug", "file": "src/app.py", "line": 1},
        "x-ai/grok-4.6",
    )
    assert ok.file == "src/app.py"
