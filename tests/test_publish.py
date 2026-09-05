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
        truncation=Truncation(
            "diff",
            truncated,
            4000 if truncated else 4,
            1000 if truncated else 4,
            1 if truncated else 300,
        ),
        mode="initial",
    )


def test_truncated_never_clean() -> None:
    assert decide_verdict(issues=[], truncated=True, successful_lanes=1) == "partial"
    assert decide_verdict(issues=[], truncated=False, successful_lanes=1) == "clean"


def test_all_lanes_failed_is_error() -> None:
    assert decide_verdict(issues=[], truncated=False, successful_lanes=0) == "error"


def test_review_discloses_requested_tier_confirmation() -> None:
    lane = LaneResult(
        SCHEMA_VERSION,
        True,
        "openai/gpt-6-astra",
        [],
        None,
        requested_service_tier="flex",
        service_tier_confirmed=True,
    )
    text = render_review(collected=_collected(), lanes=[lane], issues=[], verdict="clean")
    assert "flex tier confirmed" in text
    lane.service_tier_confirmed = False
    text = render_review(collected=_collected(), lanes=[lane], issues=[], verdict="clean")
    assert "flex tier unconfirmed" in text


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


def test_failed_lane_reports_signature_recovery_count() -> None:
    lane = LaneResult(
        SCHEMA_VERSION,
        False,
        "google/gemini-3.8-flash",
        [],
        "salvage failed",
        thought_signature_recoveries=1,
    )
    text = render_review(collected=_collected(), lanes=[lane], issues=[], verdict="error")

    assert "1 thought-signature recovery" in text


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
        MergedIssue(f"Finding number {n}", "x" * 5000, "bug", "a.py", n, ["m"]) for n in range(1, 6)
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
    # One precision per note, chosen so the breakdown visibly adds up.
    assert "**Cost:** $0.3230 (lanes $0.3223 + judge $0.0007)" in text


def test_cost_line_omitted_when_unreported() -> None:
    lane = LaneResult(SCHEMA_VERSION, True, "x-ai/grok-4.6", [], None)
    text = render_review(
        collected=_collected(),
        lanes=[lane],
        issues=[],
        verdict="clean",
    )
    assert "**Cost:** unavailable — incomplete: no cost reported for `x-ai/grok-4.6`" in text


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
    comments = inline_review_comments([issue], allowed_lines={"db.py": {9}}, generation="abc123")
    assert len(comments) == 1
    assert "\U0001f534 **Race** (`bug`)" in comments[0]["body"]


def test_incomplete_cost_totals_are_labeled() -> None:
    lane_a = LaneResult(SCHEMA_VERSION, True, "x-ai/grok-4.6", [], None, cost_usd=0.31)
    lane_b = LaneResult(SCHEMA_VERSION, True, "z-ai/glm-5.3-flash", [], None)
    text = render_review(
        collected=_collected(),
        lanes=[lane_a, lane_b],
        issues=[],
        verdict="clean",
        judge_note="`google/gemini-3.1-flash-lite`",
        judge_cost=None,
        judge_ran=True,
    )
    assert "**Cost:** $0.31" in text
    assert "incomplete: no cost reported for `z-ai/glm-5.3-flash`, the judge" in text


def test_single_known_partial_cost_renders_as_at_least_incomplete() -> None:
    lane = LaneResult(
        SCHEMA_VERSION,
        True,
        "x-ai/grok-4.6",
        [],
        None,
        known_cost_usd=0.004,
    )
    text = render_review(
        collected=_collected(),
        lanes=[lane],
        issues=[],
        verdict="clean",
    )
    assert "**Cost:** at least $0.0040 (incomplete)" in text
    assert "at least $0.0040 (incomplete)" in text


def test_mixed_complete_partial_and_judge_costs_do_not_double_count() -> None:
    lane_complete = LaneResult(SCHEMA_VERSION, True, "x-ai/grok-4.6", [], None, cost_usd=0.31)
    lane_partial = LaneResult(
        SCHEMA_VERSION,
        True,
        "z-ai/glm-5.3-flash",
        [],
        None,
        known_cost_usd=0.05,
    )
    text = render_review(
        collected=_collected(),
        lanes=[lane_complete, lane_partial],
        issues=[],
        verdict="clean",
        judge_note="`google/gemini-3.1-flash-lite`",
        judge_cost=0.01,
    )
    assert (
        "**Cost:** $0.3200 (lanes $0.3100 + judge $0.0100) + at least $0.0500 (incomplete)" in text
    )
    assert ", $0.31)" in text
    assert "at least $0.0500 (incomplete)" in text
    assert "$0.37" not in text


def test_all_unknown_costs_remain_incomplete_without_zero_total() -> None:
    lanes = [
        LaneResult(SCHEMA_VERSION, True, "x-ai/grok-4.6", [], None),
        LaneResult(SCHEMA_VERSION, True, "z-ai/glm-5.3-flash", [], None),
    ]
    text = render_review(
        collected=_collected(),
        lanes=lanes,
        issues=[],
        verdict="clean",
        judge_note="`google/gemini-3.1-flash-lite`",
        judge_cost=None,
        judge_ran=True,
    )
    assert "**Cost:** unavailable — incomplete:" in text
    assert "`x-ai/grok-4.6`" in text
    assert "`z-ai/glm-5.3-flash`" in text
    assert "the judge" in text
    assert "$0.00" not in text


def test_all_unknown_costs_without_judge_show_unavailable() -> None:
    lanes = [
        LaneResult(SCHEMA_VERSION, True, "x-ai/grok-4.6", [], None),
        LaneResult(SCHEMA_VERSION, True, "z-ai/glm-5.3-flash", [], None),
    ]
    text = render_review(
        collected=_collected(),
        lanes=lanes,
        issues=[],
        verdict="clean",
    )
    assert "**Cost:** unavailable — incomplete:" in text
    assert "`x-ai/grok-4.6`" in text
    assert "`z-ai/glm-5.3-flash`" in text
    assert "the judge" not in text
    assert "$0.00" not in text


def test_zero_partial_cost_renders_incomplete_with_and_without_judge() -> None:
    lane_partial = LaneResult(
        SCHEMA_VERSION,
        True,
        "z-ai/glm-5.3-flash",
        [],
        None,
        known_cost_usd=0.0,
    )
    without_judge = render_review(
        collected=_collected(),
        lanes=[lane_partial],
        issues=[],
        verdict="clean",
    )
    assert "**Cost:** at least $0.0000 (incomplete)" in without_judge
    assert "at least $0.0000 (incomplete)" in without_judge

    lane_complete = LaneResult(SCHEMA_VERSION, True, "x-ai/grok-4.6", [], None, cost_usd=0.31)
    with_judge = render_review(
        collected=_collected(),
        lanes=[lane_complete, lane_partial],
        issues=[],
        verdict="clean",
        judge_note="`google/gemini-3.1-flash-lite`",
        judge_cost=0.01,
    )
    assert (
        "**Cost:** $0.3200 (lanes $0.3100 + judge $0.0100) + at least $0.0000 (incomplete)"
        in with_judge
    )


def test_explicit_zero_complete_with_unknown_lane() -> None:
    lane_complete = LaneResult(SCHEMA_VERSION, True, "x-ai/grok-4.6", [], None, cost_usd=0.0)
    lane_unknown = LaneResult(SCHEMA_VERSION, True, "z-ai/glm-5.3-flash", [], None)
    text = render_review(
        collected=_collected(),
        lanes=[lane_complete, lane_unknown],
        issues=[],
        verdict="clean",
    )
    assert "**Cost:** $0.0000 — incomplete: no cost reported for `z-ai/glm-5.3-flash`" in text
    assert ", $0.0000)" in text


def _stub_collected() -> CollectedReview:
    return CollectedReview(
        pr_number=1,
        title="t",
        body="",
        head_sha="a" * 40,
        base_ref="main",
        head_ref="feat",
        plan=DiffPlan("full-pr", "full-pr", None, "a" * 40, None),
        truncation=Truncation("diff", True, 700_000, 500_000, 600, stubbed_files=("big.json",)),
        mode="initial",
    )


def test_stub_only_truncation_renders_info_note_not_partial_banner() -> None:
    lane = LaneResult(SCHEMA_VERSION, True, "x-ai/grok-4.6", [], None)
    collected = _stub_collected()
    assert not collected.truncation.forces_partial
    text = render_review(collected=collected, lanes=[lane], issues=[], verdict="clean")
    assert "Diff-budget triage" in text
    assert "partial** review" not in text
    assert "must not be treated as clean" not in text


def test_dropped_files_truncation_renders_partial_banner() -> None:
    lane = LaneResult(SCHEMA_VERSION, True, "x-ai/grok-4.6", [], None)
    collected = CollectedReview(
        pr_number=1,
        title="t",
        body="",
        head_sha="a" * 40,
        base_ref="main",
        head_ref="feat",
        plan=DiffPlan("full-pr", "full-pr", None, "a" * 40, None),
        truncation=Truncation(
            "diff",
            True,
            700_000,
            500_000,
            600,
            stubbed_files=("big.json",),
            dropped_files=("tail.py",),
        ),
        mode="initial",
    )
    assert collected.truncation.forces_partial
    text = render_review(collected=collected, lanes=[lane], issues=[], verdict="partial")
    assert "partial** review" in text
    assert "must not be treated as clean" in text
