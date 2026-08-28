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


def test_fallback_and_stale_are_partial() -> None:
    assert (
        decide_verdict(issues=[], truncated=False, successful_lanes=1, fallback=True) == "partial"
    )
    assert decide_verdict(issues=[], truncated=False, successful_lanes=1, stale=True) == "partial"


def test_render_uses_reviewed_sha_and_extra_notices() -> None:
    lane = LaneResult(SCHEMA_VERSION, True, "x-ai/grok-4.6", [], None)
    text = render_review(
        collected=_collected(),
        lanes=[lane],
        issues=[],
        verdict="partial",
        reviewed_sha="b" * 40,
        extra_notices=["The PR head advanced after this review's diff was collected."],
    )
    assert ("b" * 40) in text
    assert "head advanced" in text


def test_mentions_neutralized_in_rendered_review() -> None:
    lane = LaneResult(SCHEMA_VERSION, True, "x-ai/grok-4.6", [], None)
    issue = MergedIssue(
        "Ping @org/oncall", "cc @someone please", "bug", "a.py", 1, ["x-ai/grok-4.6"]
    )
    text = render_review(collected=_collected(), lanes=[lane], issues=[issue], verdict="issues")
    assert "@org/oncall" not in text
    assert "@\u200borg/oncall" in text
    assert "@\u200bsomeone" in text


def test_failed_lane_error_mentions_neutralized() -> None:
    lane = LaneResult(SCHEMA_VERSION, False, "x-ai/grok-4.6", [], "boom @maintainers")
    text = render_review(collected=_collected(), lanes=[lane], issues=[], verdict="error")
    assert "@\u200bmaintainers" in text


def test_long_reviews_split_into_parts() -> None:
    from or_pr_review.publish import MAX_REVIEW_BYTES, render_review_parts

    lane = LaneResult(SCHEMA_VERSION, True, "x-ai/grok-4.6", [], None)
    issues = [
        MergedIssue(f"Finding number {n}", "x" * 5000, "bug", "a.py", n, ["m"])
        for n in range(1, 26)
    ]
    parts = render_review_parts(
        collected=_collected(),
        lanes=[lane],
        issues=issues,
        verdict="issues",
        run_url="https://example.test/run",
    )
    assert len(parts) > 1
    joined = "\n".join(parts)
    for n in range(1, 26):
        assert f"— Finding number {n}" in joined
    assert all(len(part.encode("utf-8")) <= MAX_REVIEW_BYTES for part in parts)
    assert parts[1].startswith("## OpenRouter pull-request review — continued")
    assert f"Part 1 of {len(parts)}" in parts[0]
    assert "[Workflow run](https://example.test/run)" in parts[-1]
    assert "[Workflow run]" not in parts[0]


def test_inline_comments_require_in_hunk_anchors() -> None:
    from or_pr_review.publish import inline_review_comments

    issues = [
        MergedIssue("In hunk", "b", "bug", "a.py", 42, ["m"], id="r1-1"),
        MergedIssue("Out of hunk", "b", "bug", "a.py", 10, ["m"], id="r1-2"),
        MergedIssue("Out of diff", "b", "bug", "z.py", 1, ["m"], id="r1-3"),
    ]
    comments = inline_review_comments(
        issues, allowed_lines={"a.py": {40, 41, 42, 43}}, generation="1234567890ab"
    )
    # One out-of-hunk anchor would 422 the whole batched review; only the
    # in-hunk finding may post inline.
    assert [comment["line"] for comment in comments] == [42]
    assert "<!-- or-finding:1234567890ab:r1-1 -->" in comments[0]["body"]


def test_part_one_stays_within_cap_with_marker_and_round_report() -> None:
    from or_pr_review.publish import MAX_REVIEW_BYTES, render_review_parts

    lane = LaneResult(SCHEMA_VERSION, True, "x-ai/grok-4.6", [], None)
    marker = "<!-- openrouter-review-ledger:v1:" + "A" * 39_900 + " -->"
    round_lines = ["### Round 2 resolution", ""] + [
        f"- ⏳ `r1-{n}` unaddressed — **T**: " + "n" * 450 for n in range(1, 31)
    ]
    issues = [
        MergedIssue(f"Finding number {n}", "x" * 5000, "bug", "a.py", n, ["m"])
        for n in range(1, 6)
    ]
    parts = render_review_parts(
        collected=_collected(),
        lanes=[lane],
        issues=issues,
        verdict="issues",
        hidden_marker=marker,
        round_lines=round_lines,
    )
    assert all(len(part.encode("utf-8")) <= MAX_REVIEW_BYTES for part in parts)
    assert "[review body truncated]" not in "\n".join(parts)
    assert marker in parts[0]
    assert "more resolution line(s) omitted" in parts[0]
    joined = "\n".join(parts)
    for n in range(1, 6):
        assert f"— Finding number {n}" in joined


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


def test_cost_renders_on_lane_lines_and_total() -> None:
    lane_a = LaneResult(SCHEMA_VERSION, True, "x-ai/grok-4.6", [], None, cost_usd=0.31)
    lane_b = LaneResult(SCHEMA_VERSION, True, "z-ai/glm-5.3-flash", [], None, cost_usd=0.0123)
    text = render_review(
        collected=_collected(),
        lanes=[lane_a, lane_b],
        issues=[],
        verdict="clean",
        judge_note="`google/gemini-3.1-flash-lite`",
        judge_cost=0.0007,
    )
    assert ", $0.31)" in text
    assert ", $0.0123)" in text
    assert "**Cost:** $0.32 (lanes $0.32 + judge $0.0007)" in text


def test_cost_line_omitted_when_unreported() -> None:
    lane = LaneResult(SCHEMA_VERSION, True, "x-ai/grok-4.6", [], None)
    text = render_review(
        collected=_collected(),
        lanes=[lane],
        issues=[],
        verdict="clean",
    )
    assert "**Cost:**" not in text


def test_inline_comments_carry_severity_emoji() -> None:
    from or_pr_review.publish import inline_review_comments

    issue = MergedIssue(
        title="Race",
        body="check-then-act",
        severity="bug",
        file="db.py",
        line=9,
        models=["x-ai/grok-4.6"],
        id="r1-1",
    )
    comments = inline_review_comments(
        [issue], allowed_lines={"db.py": {9}}, generation="abc123"
    )
    assert len(comments) == 1
    assert "\U0001f534 **Race** (`bug`)" in comments[0]["body"]
