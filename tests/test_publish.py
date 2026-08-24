from __future__ import annotations

from or_pr_review.collect import CollectedReview, DiffPlan, Truncation
from or_pr_review.merge import MergedIssue
from or_pr_review.publish import decide_verdict, fail_on_should_fail, render_review
from or_pr_review.schema import SCHEMA_VERSION, LaneResult


def _collected(*, truncated: bool = False) -> CollectedReview:
    return CollectedReview(
        pr_number=1,
        title="t",
        body="",
        head_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        base_ref="main",
        head_ref="feat",
        plan=DiffPlan("full-pr", "full-pr", None, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", None),
        truncation=Truncation("diff", truncated, 4000 if truncated else 4, 1000 if truncated else 4, 1 if truncated else 300),
        mode="initial",
    )


def test_truncated_never_clean() -> None:
    assert decide_verdict(issues=[], truncated=True, successful_lanes=1) == "partial"
    assert decide_verdict(issues=[], truncated=False, successful_lanes=1) == "clean"


def test_all_lanes_failed_is_error() -> None:
    assert decide_verdict(issues=[], truncated=False, successful_lanes=0) == "error"


def test_fail_on_policies() -> None:
    bug = MergedIssue("a", "b", "bug", None, None, ["m"])
    nit = MergedIssue("a", "b", "nit", None, None, ["m"])
    assert not fail_on_should_fail("never", [bug])
    assert fail_on_should_fail("bugs", [bug])
    assert not fail_on_should_fail("bugs", [nit])
    assert fail_on_should_fail("any", [nit])


def test_render_includes_partial_banner() -> None:
    lane = LaneResult(SCHEMA_VERSION, True, "x-ai/grok-4.6", [], None)
    text = render_review(
        collected=_collected(truncated=True),
        lanes=[lane],
        issues=[],
        verdict="partial",
    )
    assert "partial" in text.lower()
    assert "must not be treated as a clean review" in text
