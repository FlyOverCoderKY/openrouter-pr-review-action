from __future__ import annotations

from or_pr_review.merge import MergedIssue, deduplicate_issues, issues_from_single_lane
from or_pr_review.schema import SCHEMA_VERSION, Finding, LaneResult


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


def test_application_environment_language_is_not_a_review_diagnostic() -> None:
    lane = _ok(
        "m1",
        [
            _finding(
                "Secret not loaded in the current environment",
                model="m1",
                body=(
                    "The application could not find the key when the current "
                    "environment variable is unavailable in staging."
                ),
                file="src/config.py",
                line=None,
            ),
            _finding(
                "This workspace helper cannot find generated state",
                model="m1",
                body="The workspace helper returns unavailable for a valid project.",
                file="src/workspace.py",
                line=None,
            ),
        ],
    )

    assert [issue.title for issue in issues_from_single_lane(lane)] == [
        "Secret not loaded in the current environment",
        "This workspace helper cannot find generated state",
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
