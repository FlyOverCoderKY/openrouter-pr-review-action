from __future__ import annotations

from or_pr_review.merge import MergedIssue, format_issue_block, identified_by


def test_identified_by_one_two_and_many() -> None:
    assert identified_by(["x-ai/grok-4.6"]) == "identified by x-ai/grok-4.6"
    assert (
        identified_by(["x-ai/grok-4.6", "anthropic/claude-sonnet-4.6"])
        == "identified by x-ai/grok-4.6 and anthropic/claude-sonnet-4.6"
    )
    assert identified_by(
        ["x-ai/grok-4.6", "anthropic/claude-sonnet-4.6", "google/gemini-2.5-pro"]
    ) == (
        "identified by x-ai/grok-4.6, anthropic/claude-sonnet-4.6, and google/gemini-2.5-pro"
    )


def test_heading_and_body_format() -> None:
    issue = MergedIssue(
        title="Missing auth check",
        body="The handler accepts unauthenticated POSTs.",
        severity="bug",
        file="src/api.py",
        line=42,
        models=["x-ai/grok-4.6", "anthropic/claude-sonnet-4.6"],
    )
    block = format_issue_block(1, issue)
    assert block.startswith("#### \U0001f534 Issue 1 — Missing auth check\n")
    assert (
        "`src/api.py:42` · `bug` · identified by x-ai/grok-4.6 "
        "and anthropic/claude-sonnet-4.6"
    ) in block
    assert "The handler accepts unauthenticated POSTs." in block


def test_heading_preserves_issue_number() -> None:
    issue = MergedIssue(
        title="Docs typo",
        body="teh",
        severity="nit",
        file=None,
        line=None,
        models=["x-ai/grok-4.6"],
    )
    assert issue.heading(3) == "Issue 3 - Docs typo (identified by x-ai/grok-4.6)"
