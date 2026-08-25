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
