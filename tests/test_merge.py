from __future__ import annotations

from or_pr_review.merge import MergedIssue, deduplicate_issues, issues_from_single_lane, merge_lanes
from or_pr_review.schema import SCHEMA_VERSION, Finding, LaneResult, failed_lane


def _finding(
    title: str,
    *,
    model: str,
    body: str = "details",
    severity: str = "bug",
    file: str | None = "a.py",
    line: int | None = 1,
) -> Finding:
    return Finding(
        title=title,
        body=body,
        severity=severity,
        file=file,
        line=line,
        model_id=model,
    )


def _ok(model: str, findings: list[Finding]) -> LaneResult:
    return LaneResult(
        schema_version=SCHEMA_VERSION,
        ok=True,
        model=model,
        findings=findings,
        error=None,
    )


def test_merge_dedupes_same_title_and_file() -> None:
    lanes = [
        _ok(
            "x-ai/grok-4.6",
            [_finding("Missing auth check", model="x-ai/grok-4.6", body="short")],
        ),
        _ok(
            "anthropic/claude-sonnet-4.6",
            [
                _finding(
                    "Missing auth check",
                    model="anthropic/claude-sonnet-4.6",
                    body="a longer explanation of the same bug",
                )
            ],
        ),
    ]
    merged = merge_lanes(lanes)
    assert len(merged) == 1
    assert merged[0].models == ["x-ai/grok-4.6", "anthropic/claude-sonnet-4.6"]
    assert "longer explanation" in merged[0].body


def test_merge_keeps_distinct_titles() -> None:
    lanes = [
        _ok("x-ai/grok-4.6", [_finding("Auth", model="x-ai/grok-4.6")]),
        _ok(
            "anthropic/claude-sonnet-4.6",
            [_finding("Off-by-one", model="anthropic/claude-sonnet-4.6")],
        ),
    ]
    assert len(merge_lanes(lanes)) == 2


def test_merge_similar_titles_same_file() -> None:
    lanes = [
        _ok("m1", [_finding("SQL injection in query builder", model="m1", file="core.c")]),
        _ok("m2", [_finding("SQL injection in query", model="m2", file="core.c")]),
    ]
    merged = merge_lanes(lanes)
    assert len(merged) == 1
    assert merged[0].models == ["m1", "m2"]


def test_failed_lanes_are_ignored() -> None:
    lanes = [
        failed_lane("x-ai/grok-4.6", "timeout"),
        _ok(
            "anthropic/claude-sonnet-4.6",
            [_finding("Only from B", model="anthropic/claude-sonnet-4.6")],
        ),
    ]
    merged = merge_lanes(lanes)
    assert len(merged) == 1
    assert merged[0].models == ["anthropic/claude-sonnet-4.6"]


def test_severity_promotes_to_bug() -> None:
    lanes = [
        _ok("m1", [_finding("Same", model="m1", severity="nit")]),
        _ok("m2", [_finding("Same", model="m2", severity="bug")]),
    ]
    assert merge_lanes(lanes)[0].severity == "bug"


def test_single_lane_posts_without_dedupe() -> None:
    lane = _ok(
        "x-ai/grok-4.6",
        [
            _finding("Same title", model="x-ai/grok-4.6", body="first"),
            _finding("Same title", model="x-ai/grok-4.6", body="second"),
        ],
    )
    issues = issues_from_single_lane(lane)
    assert len(issues) == 2
    assert all(issue.models == ["x-ai/grok-4.6"] for issue in issues)


def test_bugs_sort_before_nits() -> None:
    lanes = [
        _ok(
            "m1",
            [
                _finding("Nitty", model="m1", severity="nit", file="z.py"),
                _finding("Broken", model="m1", severity="bug", file="a.py"),
            ],
        )
    ]
    merged = merge_lanes(lanes)
    assert [issue.severity for issue in merged] == ["bug", "nit"]


def test_single_lane_keeps_checkout_access_failure_as_diagnostic_only(capsys) -> None:
    lane = _ok(
        "z-ai/glm-5.3-flash",
        [
            _finding(
                "The edited registry file is not present/readable in the review checkout",
                model="z-ai/glm-5.3-flash",
                body=(
                    "The diff edits the registry, but that file cannot be opened or "
                    "searched in this checkout. Its contents are therefore unverified."
                ),
                file="src/rules/registry.ts",
                line=None,
                severity="risk",
            ),
            _finding("Real defect", model="z-ai/glm-5.3-flash", line=8),
        ],
    )

    issues = issues_from_single_lane(lane)

    assert [issue.title for issue in issues] == ["Real defect"]
    assert "not published as a code finding" in capsys.readouterr().out


def test_missing_file_defect_is_not_mistaken_for_review_environment_failure() -> None:
    lane = _ok(
        "m1",
        [
            _finding(
                "Generated manifest is missing",
                model="m1",
                body=(
                    "The application opens manifest.json at startup, but this PR never creates it."
                ),
                file="src/startup.py",
                line=None,
            )
        ],
    )

    assert [issue.title for issue in issues_from_single_lane(lane)] == [
        "Generated manifest is missing"
    ]


def test_deduplicate_issues_absorbs_conservative_near_match() -> None:
    issues = [
        MergedIssue(
            title=(
                "Both new fixtures largely re-pin neighboring tests and include "
                "tautological not-assertions"
            ),
            body=(
                "Both fixtures repeat neighboring assertions and add tautological negative checks."
            ),
            severity="nit",
            file="src/filingEvidence.test.ts",
            line=284,
            models=["m1"],
        ),
        MergedIssue(
            title=(
                "Both new fixtures re-pin neighboring tests and include tautological not assertions"
            ),
            body=(
                "Both fixtures repeat the neighboring assertions and add tautological "
                "negative checks."
            ),
            severity="risk",
            file="src/filingEvidence.test.ts",
            line=284,
            models=["m2"],
        ),
    ]

    deduped, absorbed = deduplicate_issues(issues)

    assert absorbed == 1
    assert len(deduped) == 1
    assert deduped[0].severity == "risk"
    assert deduped[0].models == ["m1", "m2"]


def test_deduplicate_issues_preserves_distinct_defects_at_same_location() -> None:
    issues = [
        MergedIssue("Missing auth", "POST is unauthenticated", "bug", "api.py", 8, ["m1"]),
        MergedIssue("Lost update", "Write is not atomic", "bug", "api.py", 8, ["m2"]),
    ]

    deduped, absorbed = deduplicate_issues(issues)

    assert absorbed == 0
    assert len(deduped) == 2


def test_deduplicate_issues_preserves_same_title_without_location_or_matching_body() -> None:
    issues = [
        MergedIssue("Missing validation", "Email is unchecked", "bug", None, None, ["m1"]),
        MergedIssue("Missing validation", "Age is unchecked", "bug", None, None, ["m2"]),
    ]

    deduped, absorbed = deduplicate_issues(issues)

    assert absorbed == 0
    assert len(deduped) == 2


def test_deduplicate_issues_preserves_same_title_and_location_with_different_evidence() -> None:
    issues = [
        MergedIssue(
            "Missing validation",
            "The endpoint accepts a negative age.",
            "bug",
            "src/api.py",
            42,
            ["m1"],
        ),
        MergedIssue(
            "Missing validation",
            "The endpoint accepts an unauthorized account ID.",
            "bug",
            "src/api.py",
            42,
            ["m2"],
        ),
    ]

    deduped, absorbed = deduplicate_issues(issues)

    assert absorbed == 0
    assert len(deduped) == 2
