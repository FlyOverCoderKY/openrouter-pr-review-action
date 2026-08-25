"""Build the posted review body and apply fail_on."""

from __future__ import annotations

from or_pr_review.collect import CollectedReview
from or_pr_review.merge import MergedIssue, format_issue_block
from or_pr_review.schema import LaneResult

MAX_REVIEW_CHARS = 60_000


def decide_verdict(
    *,
    issues: list[MergedIssue],
    truncated: bool,
    successful_lanes: int,
    fallback: bool = False,
    stale: bool = False,
) -> str:
    """`fallback` (single-commit diff fallback) and `stale` (PR head advanced
    past the reviewed commit) are partial, never clean: both mean this review
    did not see everything a clean verdict would vouch for."""
    if successful_lanes == 0:
        return "error"
    if truncated or fallback or stale:
        return "partial"
    return "issues" if issues else "clean"


def fail_on_should_fail(fail_on: str, issues: list[MergedIssue]) -> bool:
    policy = (fail_on or "never").strip().lower()
    if policy == "never":
        return False
    if policy == "bugs":
        return any(issue.severity == "bug" for issue in issues)
    if policy == "any":
        return bool(issues)
    raise ValueError(f"fail_on must be never, bugs, or any; got {fail_on!r}")


def render_review(
    *,
    collected: CollectedReview,
    lanes: list[LaneResult],
    issues: list[MergedIssue],
    verdict: str,
    run_url: str = "",
    judge_note: str = "",
    reviewed_sha: str | None = None,
    extra_notices: list[str] | None = None,
) -> str:
    lane_lines = []
    for lane in lanes:
        if lane.ok:
            extra = f"{len(lane.findings)} finding(s)"
            if lane.elapsed_ms is not None:
                extra += f", {lane.elapsed_ms / 1000:.1f}s"
            if lane.prompt_tokens is not None:
                extra += f", {lane.prompt_tokens}+{lane.completion_tokens or 0} tokens"
                if lane.cached_tokens:
                    extra += f" ({lane.cached_tokens} cached)"
            if lane.tool_rounds is not None:
                extra += f", {lane.tool_rounds} tool round(s)"
            if lane.retries:
                extra += f", {lane.retries} retried request(s)"
            if lane.salvaged:
                extra += ", salvaged finish"
            lane_lines.append(f"- `{lane.model}`: ok ({extra})")
        else:
            lane_lines.append(f"- `{lane.model}`: failed-open — {lane.error or 'unknown error'}")

    header = [
        "## OpenRouter pull-request review",
        "",
        f"**Verdict:** `{verdict}`",
        f"**Scope:** `{collected.plan.scope}` ({collected.plan.kind})",
        f"**Mode:** `{collected.mode}`",
        f"**Commit:** `{reviewed_sha or collected.head_sha}`",
    ]
    if judge_note:
        header.append(f"**Judge:** {judge_note}")
    header.extend(
        [
            "",
            "### Lanes",
            "",
            *lane_lines,
            "",
        ]
    )
    if collected.truncation.truncated and collected.truncation.notice:
        header.extend(
            [
                "> This is a **partial** review. The embedded diff was truncated. "
                "It must not be treated as a clean review.",
                "",
                collected.truncation.notice,
                "",
            ]
        )
    if collected.plan.fallback_notice:
        header.extend([collected.plan.fallback_notice, ""])
    for notice in extra_notices or []:
        header.extend(["> [!WARNING]", f"> {notice}", ""])

    if issues:
        header.append("### Findings")
        header.append("")
        for index, issue in enumerate(issues, start=1):
            header.append(format_issue_block(index, issue))
            header.append("")
    else:
        header.append("No structured findings from the successful lane(s).")
        header.append("")

    if run_url:
        header.extend([f"[Workflow run]({run_url})", ""])

    text = "\n".join(header).rstrip() + "\n"
    if len(text) > MAX_REVIEW_CHARS:
        text = text[: MAX_REVIEW_CHARS - 80].rstrip() + "\n\n[review body truncated]\n"
    return text


def render_incomplete(*, stage: str, reason: str, run_url: str = "") -> str:
    lines = [
        "## OpenRouter review incomplete",
        "",
        f"The action stopped during **{stage}**.",
        "",
        reason,
        "",
    ]
    if run_url:
        lines.extend([f"[Workflow run]({run_url})", ""])
    return "\n".join(lines)
