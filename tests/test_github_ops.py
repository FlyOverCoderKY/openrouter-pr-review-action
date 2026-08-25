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


def test_compare_diff_rejects_non_fast_forward() -> None:
    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        return json.dumps({"status": "diverged", "behind_by": 3})

    gh = GitHub(token="t", repository="o/r", runner=runner)
    with pytest.raises(ActionError, match="fast-forward"):
        gh.compare_diff("a" * 40, "b" * 40)


def test_commit_diff_uses_raw_media_type() -> None:
    calls: list[list[str]] = []

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        calls.append(cmd)
        return "raw-commit-diff"

    gh = GitHub(token="t", repository="o/r", runner=runner)
    assert gh.commit_diff("c" * 40) == "raw-commit-diff"
    assert "Accept: application/vnd.github.diff" in calls[0]


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


def test_list_finding_replies_pairs_marker_threads() -> None:
    comments = [
        [
            {"id": 10, "body": "<!-- or-finding:r1-1 -->\n**T**", "user": {"login": "bot"}},
            {"id": 11, "in_reply_to_id": 10, "body": "fixed in abc123", "user": {"login": "dev"}},
            {"id": 12, "in_reply_to_id": 99, "body": "unrelated", "user": {"login": "dev"}},
        ]
    ]

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        return json.dumps(comments)

    gh = GitHub(token="t", repository="o/r", runner=runner)
    assert gh.list_finding_replies(1) == [("r1-1", "dev", "fixed in abc123")]
