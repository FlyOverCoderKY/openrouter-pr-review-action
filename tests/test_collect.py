from __future__ import annotations

import pytest

from or_pr_review.collect import (
    COMPARE_FAILED_NOTICE,
    MISSING_BEFORE_NOTICE,
    collect_review,
    fetch_scoped_diff,
    plan_diff,
    resolve_mode,
    truncate_diff,
)
from or_pr_review.errors import ActionError


class FakeSource:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.compare_error: Exception | None = None

    def pr_view(self, number: int) -> dict[str, object]:
        self.calls.append(("pr_view", number))
        return {
            "title": "Add login",
            "body": "please review",
            "headRefOid": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "headRefName": "feature",
            "baseRefName": "main",
        }

    def pr_diff(self, number: int) -> str:
        self.calls.append(("pr_diff", number))
        return "FULL_PR_DIFF"

    def compare_diff(self, before: str, after: str) -> str:
        self.calls.append(("compare_diff", (before, after)))
        if self.compare_error:
            raise self.compare_error
        return "RANGE_DIFF"

    def commit_diff(self, sha: str) -> str:
        self.calls.append(("commit_diff", sha))
        return "SINGLE_COMMIT_DIFF"


def test_latest_commit_plans_range_not_full_pr() -> None:
    plan = plan_diff(
        scope="latest-commit",
        before_sha="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        after_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        head_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    assert plan.kind == "commit-range"
    assert plan.kind != "full-pr"
    assert plan.scope == "latest-commit"


def test_latest_commit_missing_before_is_single_commit_not_full_pr() -> None:
    plan = plan_diff(
        scope="latest-commit",
        before_sha=None,
        after_sha=None,
        head_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    assert plan.kind == "single-commit"
    assert plan.fallback_notice == MISSING_BEFORE_NOTICE


def test_latest_commit_without_head_refuses_full_pr() -> None:
    with pytest.raises(ActionError, match="Refusing to fall back to the full PR diff"):
        plan_diff(scope="latest-commit", before_sha=None, after_sha=None, head_sha=None)


def test_fetch_latest_commit_never_calls_pr_diff() -> None:
    source = FakeSource()
    plan = plan_diff(
        scope="latest-commit",
        before_sha="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        after_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        head_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    diff, used = fetch_scoped_diff(7, plan, source)
    assert diff == "RANGE_DIFF"
    assert used.kind == "commit-range"
    assert ("pr_diff", 7) not in source.calls


def test_fetch_latest_commit_compare_failure_falls_back_to_single_commit() -> None:
    source = FakeSource()
    source.compare_error = ActionError("compare failed")
    plan = plan_diff(
        scope="latest-commit",
        before_sha="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        after_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        head_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    diff, used = fetch_scoped_diff(7, plan, source)
    assert diff == "SINGLE_COMMIT_DIFF"
    assert used.kind == "single-commit"
    assert used.fallback_notice == COMPARE_FAILED_NOTICE
    assert not any(name == "pr_diff" for name, _ in source.calls)


def test_truncated_diff_is_partial_and_not_clean() -> None:
    huge = "x" * 4000
    truncation = truncate_diff(huge, max_diff_kb=1)
    assert truncation.truncated
    assert truncation.notice is not None
    assert "must not be treated as clean" in truncation.notice
    assert truncation.embedded_bytes <= 1024


def test_auto_mode_maps_events() -> None:
    assert resolve_mode("auto", "opened") == "initial"
    assert resolve_mode("auto", "synchronize") == "verify"
    assert resolve_mode("initial", "synchronize") == "initial"
    assert resolve_mode("verify", "opened") == "verify"


def test_initial_mode_rejects_latest_commit_scope() -> None:
    with pytest.raises(ActionError, match="initial review_mode requires review_scope=full-pr"):
        collect_review(
            pr_number=1,
            scope="latest-commit",
            mode="initial",
            before_sha=None,
            after_sha=None,
            head_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            max_diff_kb=300,
            source=FakeSource(),
        )


def test_fetch_scoped_diff_distinguishes_diverged_from_transport() -> None:
    from or_pr_review.collect import (
        COMPARE_FAILED_NOTICE,
        DIVERGED_NOTICE,
        DiffPlan,
        fetch_scoped_diff,
    )
    from or_pr_review.errors import ActionError, DivergedRangeError

    plan = DiffPlan("latest-commit", "commit-range", "a" * 40, "b" * 40, None)

    class _Source:
        def __init__(self, exc: Exception) -> None:
            self.exc = exc

        def pr_view(self, number: int) -> dict[str, object]:
            raise AssertionError("unused")

        def pr_diff(self, number: int) -> str:
            raise AssertionError("unused")

        def compare_diff(self, before: str, after: str) -> str:
            raise self.exc

        def commit_diff(self, sha: str) -> str:
            return "single-commit-diff"

    _diff, diverged_plan = fetch_scoped_diff(1, plan, _Source(DivergedRangeError("nff")))
    assert diverged_plan.fallback_notice == DIVERGED_NOTICE
    _diff, transport_plan = fetch_scoped_diff(1, plan, _Source(ActionError("timeout")))
    assert transport_plan.fallback_notice == COMPARE_FAILED_NOTICE


def test_collect_review_packs_over_budget_diff_with_triage_inputs() -> None:
    handwritten = (
        "diff --git a/src/main.py b/src/main.py\n"
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+alpha\n"
        "+beta\n"
    )
    snapshot = (
        "diff --git a/src/data/snap.dat b/src/data/snap.dat\n"
        "--- a/src/data/snap.dat\n"
        "+++ b/src/data/snap.dat\n"
        "@@ -0,0 +1,40 @@\n" + "".join(f"+{'y' * 60}\n" for _ in range(40))
    )

    class _Source(FakeSource):
        def pr_diff(self, number: int) -> str:
            return handwritten + snapshot

    collected = collect_review(
        pr_number=1,
        scope="full-pr",
        mode="initial",
        before_sha=None,
        after_sha=None,
        head_sha=None,
        max_diff_kb=1,
        source=_Source(),
        gitattributes_text="src/data/** linguist-generated\n",
    )
    assert collected.truncation.truncated
    assert collected.truncation.stubbed_files == ("src/data/snap.dat",)
    assert not collected.truncation.forces_partial
    assert handwritten in collected.diff
    # Full-diff path accounting is unaffected by the packing.
    assert collected.all_changed_paths == ("src/main.py", "src/data/snap.dat")
