"""Build the posted review body and apply fail_on."""

from __future__ import annotations

from or_pr_review.collect import CollectedReview
from or_pr_review.merge import MergedIssue, format_issue_block, neutralize_mentions
from or_pr_review.schema import LaneResult

# GitHub caps comment bodies at 65,536 characters; stay under it in bytes and
# continue long findings lists in follow-up comments instead of dropping them.
MAX_REVIEW_BYTES = 60_000
TARGET_REVIEW_BYTES = 58_000
_CONTINUED_HEADING = "## OpenRouter pull-request review — continued"


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
    """Single-body rendering; the first part of render_review_parts."""
    return render_review_parts(
        collected=collected,
        lanes=lanes,
        issues=issues,
        verdict=verdict,
        run_url=run_url,
        judge_note=judge_note,
        reviewed_sha=reviewed_sha,
        extra_notices=extra_notices,
    )[0]


def render_review_parts(
    *,
    collected: CollectedReview,
    lanes: list[LaneResult],
    issues: list[MergedIssue],
    verdict: str,
    run_url: str = "",
    judge_note: str = "",
    reviewed_sha: str | None = None,
    extra_notices: list[str] | None = None,
) -> list[str]:
    """The review body plus continuation-comment bodies for long findings
    lists, so nothing is dropped to fit GitHub's body limit."""
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
            lane_lines.append(
                f"- `{lane.model}`: failed-open — "
                f"{neutralize_mentions(lane.error or 'unknown error')}"
            )

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

    if not issues:
        lines = [*header, "No structured findings from the successful lane(s).", ""]
        if run_url:
            lines.extend([f"[Workflow run]({run_url})", ""])
        return [_finalize(lines)]

    header.extend(["### Findings", ""])
    part_lines: list[list[str]] = []
    current = header
    prefix_len = len(current)
    for index, issue in enumerate(issues, start=1):
        block = [format_issue_block(index, issue), ""]
        candidate = "\n".join([*current, *block])
        if len(candidate.encode("utf-8")) > TARGET_REVIEW_BYTES and len(current) > prefix_len:
            part_lines.append(current)
            current = [_CONTINUED_HEADING, "", "### Findings (continued)", ""]
            prefix_len = len(current)
        current.extend(block)
    part_lines.append(current)

    total = len(part_lines)
    bodies: list[str] = []
    for number, lines in enumerate(part_lines, start=1):
        if total > 1:
            lines = [*lines, f"_Part {number} of {total}; all findings are preserved._", ""]
        if number == total and run_url:
            lines = [*lines, f"[Workflow run]({run_url})", ""]
        bodies.append(_finalize(lines))
    return bodies


def _finalize(lines: list[str]) -> str:
    text = "\n".join(lines).rstrip() + "\n"
    if len(text.encode("utf-8")) > MAX_REVIEW_BYTES:
        clipped = text.encode("utf-8")[: MAX_REVIEW_BYTES - 80].decode("utf-8", errors="ignore")
        text = clipped.rstrip() + "\n\n[review body truncated]\n"
    return text


def render_incomplete(*, stage: str, reason: str, run_url: str = "") -> str:
    lines = [
        "## OpenRouter review incomplete",
        "",
        f"The action stopped during **{stage}**.",
        "",
        neutralize_mentions(reason),
        "",
    ]
    if run_url:
        lines.extend([f"[Workflow run]({run_url})", ""])
    return "\n".join(lines)
