from __future__ import annotations

import json

import pytest

from or_pr_review.errors import ActionError
from or_pr_review.github_ops import GitHub


def test_compare_diff_validates_fast_forward_and_fetches_raw() -> None:
    calls: list[list[str]] = []

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        calls.append(cmd)
        if "-H" in cmd:
            return "raw-diff-text"
        return json.dumps({"status": "ahead", "behind_by": 0})

    gh = GitHub(token="t", repository="o/r", runner=runner)
    assert gh.compare_diff("a" * 40, "b" * 40) == "raw-diff-text"
    assert len(calls) == 2
    assert "Accept: application/vnd.github.diff" in calls[1]


def test_compare_diff_rejects_non_fast_forward_with_distinct_error() -> None:
    from or_pr_review.errors import DivergedRangeError

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        return json.dumps({"status": "diverged", "behind_by": 3})

    gh = GitHub(token="t", repository="o/r", runner=runner)
    # The distinct type matters: only a genuine divergence may reset the
    # review loop; transport failures raise plain ActionError.
    with pytest.raises(DivergedRangeError, match="fast-forward"):
        gh.compare_diff("a" * 40, "b" * 40)


def test_commit_diff_uses_raw_media_type() -> None:
    calls: list[list[str]] = []

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        calls.append(cmd)
        return "raw-commit-diff"

    gh = GitHub(token="t", repository="o/r", runner=runner)
    assert gh.commit_diff("c" * 40) == "raw-commit-diff"
    assert "Accept: application/vnd.github.diff" in calls[0]


@pytest.mark.parametrize(
    "too_large_message",
    [
        "HTTP 406: Sorry, the diff exceeded the maximum number of lines (20000)",
        "HTTP 406: PullRequest.diff too_large",
    ],
)
def test_pr_diff_falls_back_to_local_git_on_github_line_limit(
    monkeypatch, too_large_message: str
) -> None:
    base = "a" * 40
    head = "b" * 40
    gh_calls: list[list[str]] = []
    git_calls: list[tuple[list[str], int, str]] = []

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        gh_calls.append(cmd)
        assert cmd[1:3] == ["pr", "diff"]
        raise ActionError(too_large_message)

    def git_runner(cmd: list[str], *, timeout: int, cwd: str) -> str:
        git_calls.append((cmd, timeout, cwd))
        return "complete-local-diff"

    gh = GitHub(
        token="t",
        repository="o/r",
        timeout=45,
        runner=runner,
        git_runner=git_runner,
        source_workspace="/checkout",
    )
    assert gh.pr_diff(7, base_sha=base, head_sha=head) == "complete-local-diff"
    assert len(gh_calls) == 1
    assert git_calls == [
        (
            [
                "git",
                "diff",
                "--no-color",
                "--no-ext-diff",
                "--no-textconv",
                "--find-renames",
                f"{base}...{head}",
                "--",
            ],
            45,
            "/checkout",
        )
    ]


def test_pr_diff_does_not_hide_other_github_failures() -> None:
    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        raise ActionError("HTTP 403: resource not accessible")

    def git_runner(cmd: list[str], *, timeout: int, cwd: str) -> str:
        raise AssertionError("local git must not run")

    gh = GitHub(token="t", repository="o/r", runner=runner, git_runner=git_runner)
    with pytest.raises(ActionError, match="HTTP 403"):
        gh.pr_diff(7)


def test_list_bot_review_bodies_filters_author() -> None:
    reviews = [
        [
            {"user": {"login": "github-actions[bot]"}, "body": "mine"},
            {"user": {"login": "someone"}, "body": "not mine"},
            {"user": {"login": "github-actions[bot]"}, "body": None},
        ]
    ]

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        assert "--paginate" in cmd and "--slurp" in cmd
        return json.dumps(reviews)

    gh = GitHub(token="t", repository="o/r", runner=runner)
    assert gh.list_bot_review_bodies(1, "github-actions[bot]") == ["mine"]


def test_create_review_falls_back_without_comments() -> None:
    calls: list[dict] = []

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        calls.append(json.loads(stdin or "{}"))
        if len(calls) == 1:
            raise ActionError("422 line is not part of the diff")
        return json.dumps({"html_url": "https://example.test/review"})

    gh = GitHub(token="t", repository="o/r", runner=runner)
    result = gh.create_review(
        1,
        "body",
        "c" * 40,
        comments=[{"path": "a.py", "line": 1, "side": "RIGHT", "body": "x"}],
    )
    assert result["html_url"] == "https://example.test/review"
    assert "comments" in calls[0]
    assert "comments" not in calls[1]


def test_create_review_does_not_retry_after_transport_failure() -> None:
    calls: list[dict] = []

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        calls.append(json.loads(stdin or "{}"))
        raise ActionError("GitHub CLI timed out after 120s")

    gh = GitHub(token="t", repository="o/r", runner=runner)
    with pytest.raises(ActionError, match="timed out"):
        gh.create_review(
            1,
            "body",
            "c" * 40,
            comments=[{"path": "a.py", "line": 1, "side": "RIGHT", "body": "x"}],
        )
    assert len(calls) == 1


def test_upsert_status_comment_uses_paginated_issue_comments() -> None:
    from or_pr_review.github_ops import STATUS_MARKER, upsert_status_comment

    calls: list[list[str]] = []

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        calls.append(cmd)
        if "--paginate" in cmd:
            return json.dumps([[{"id": 101, "body": f"{STATUS_MARKER}\nold"}]])
        return json.dumps({"html_url": "https://example.test/status"})

    gh = GitHub(token="t", repository="o/r", runner=runner)
    assert upsert_status_comment(gh, pr_number=1, body="new") == "https://example.test/status"
    assert "--paginate" in calls[0]
    assert "PATCH" in calls[1]


def test_list_finding_replies_pairs_marker_threads_by_generation() -> None:
    generation = "1234567890ab"
    other_generation = "feedfeedfeed"
    comments = [
        [
            {
                "id": 10,
                "body": f"<!-- or-finding:{generation}:r1-1 -->\n**T**",
                "user": {"login": "bot"},
            },
            {"id": 11, "in_reply_to_id": 10, "body": "fixed in abc123", "user": {"login": "dev"}},
            {
                "id": 20,
                "body": f"<!-- or-finding:{other_generation}:r1-1 -->\n**Old**",
                "user": {"login": "bot"},
            },
            {
                "id": 21,
                "in_reply_to_id": 20,
                "body": "false positive because X",
                "user": {"login": "dev"},
            },
            {"id": 12, "in_reply_to_id": 99, "body": "unrelated", "user": {"login": "dev"}},
        ]
    ]

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        return json.dumps(comments)

    gh = GitHub(token="t", repository="o/r", runner=runner)
    # Only the current generation's thread pairs; the pre-reset thread with a
    # reused finding id is ignored, and an empty generation harvests nothing.
    assert gh.list_finding_replies(1, generation=generation) == [("r1-1", "dev", "fixed in abc123")]
    assert gh.list_finding_replies(1, generation="") == []


def test_recent_issue_comments_exclude_bot_review_bodies() -> None:
    comments = [
        [
            {
                "id": 1,
                "body": "## OpenRouter pull-request review — continued\n...",
                "user": {"login": "b"},
            },
            {"id": 2, "body": "## OpenRouter review incomplete\n...", "user": {"login": "b"}},
            {"id": 3, "body": "I pushed a fix for r1-1", "user": {"login": "dev"}},
        ]
    ]

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        return json.dumps(comments)

    gh = GitHub(token="t", repository="o/r", runner=runner)
    assert gh.list_recent_issue_comments(1) == [("dev", "I pushed a fix for r1-1")]
