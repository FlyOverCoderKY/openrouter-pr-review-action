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


def test_coverage_schema_flags() -> None:
    from or_pr_review.schema import findings_json_schema

    base = findings_json_schema()
    assert "coverage" not in base["schema"]["properties"]
    cov = findings_json_schema(include_coverage=True)
    assert "coverage" in cov["schema"]["properties"]
    assert "coverage" in cov["schema"]["required"]
    res = findings_json_schema(include_resolutions=True)
    assert "resolutions" in res["schema"]["required"]
    assert "coverage" not in findings_json_schema()["schema"]["properties"]


def test_parse_lane_payload_coverage_rules() -> None:
    import pytest

    from or_pr_review.errors import LaneError
    from or_pr_review.schema import parse_lane_payload

    text = '{"findings": [], "coverage": [{"path": "a.py", "findings": 0}]}'
    findings, resolutions, coverage = parse_lane_payload(text, "m", expect_coverage=True)
    assert findings == [] and resolutions == []
    assert coverage == [("a.py", 0)]
    with pytest.raises(LaneError, match="coverage is missing"):
        parse_lane_payload('{"findings": []}', "m", expect_coverage=True)
    with pytest.raises(LaneError, match="safe relative path"):
        parse_lane_payload(
            '{"findings": [], "coverage": [{"path": "../x", "findings": 0}]}',
            "m",
            expect_coverage=True,
        )
    with pytest.raises(LaneError, match="more than once"):
        parse_lane_payload(
            '{"findings": [], "coverage": '
            '[{"path": "a.py", "findings": 0}, {"path": "a.py", "findings": 1}]}',
            "m",
            expect_coverage=True,
        )


def test_parse_lane_payload_resolutions() -> None:
    import pytest

    from or_pr_review.errors import LaneError
    from or_pr_review.schema import parse_lane_payload

    text = '{"findings": [], "resolutions": [{"id": "r1-1", "status": "Fixed", "note": "done"}]}'
    _findings, resolutions, _coverage = parse_lane_payload(text, "m", expect_resolutions=True)
    assert resolutions[0].id == "r1-1"
    assert resolutions[0].status == "fixed"
    with pytest.raises(LaneError, match="resolutions are missing"):
        parse_lane_payload('{"findings": []}', "m", expect_resolutions=True)


def test_validate_coverage_and_mismatches() -> None:
    from or_pr_review.schema import Finding, coverage_count_mismatches, validate_coverage

    assert validate_coverage([("a.py", 1)], {"a.py"}) is None
    assert "does not account" in validate_coverage([("a.py", 1)], {"a.py", "b.py"})
    assert "not in the embedded diff" in validate_coverage(
        [("a.py", 1), ("zz.py", 0)], {"a.py"}
    )
    findings = [Finding("t", "b", "bug", "a.py", 1, "m")]
    notes = coverage_count_mismatches(findings, [("a.py", 3)], {"a.py"})
    assert notes and "claims 3" in notes[0]
    assert coverage_count_mismatches(findings, [("a.py", 1)], {"a.py"}) == []


def test_lane_artifact_roundtrip_with_coverage_and_resolutions() -> None:
    from or_pr_review.schema import SCHEMA_VERSION, LaneResult, Resolution, parse_lane_artifact

    lane = LaneResult(
        schema_version=SCHEMA_VERSION,
        ok=True,
        model="x-ai/grok-4.6",
        findings=[],
        error=None,
        resolutions=[Resolution(id="r1-1", status="fixed", note="done")],
        coverage=[("a.py", 0)],
    )
    parsed = parse_lane_artifact(lane.to_dict())
    assert parsed.resolutions[0].id == "r1-1"
    assert parsed.coverage == [("a.py", 0)]
