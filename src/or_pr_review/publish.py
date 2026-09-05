"""Build the posted review body and apply fail_on."""

from __future__ import annotations

from typing import Any

from or_pr_review.collect import CollectedReview
from or_pr_review.loop import finding_marker
from or_pr_review.merge import (
    MergedIssue,
    format_issue_block,
    neutralize_mentions,
    severity_emoji,
)
from or_pr_review.schema import LaneResult

# GitHub caps comment bodies at 65,536 characters; stay under it in bytes and
# continue long findings lists in follow-up comments instead of dropping them.
MAX_REVIEW_BYTES = 60_000
TARGET_REVIEW_BYTES = 58_000
# The verify-round resolution report lives in the part-1 header alongside the
# ledger marker (≤40KB); bounding it keeps part 1 under MAX_REVIEW_BYTES so
# _finalize never has to truncate real content.
ROUND_REPORT_BYTES = 8_000
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


def fail_on_should_fail(
    fail_on: str,
    issues: list[MergedIssue],
    *,
    open_issue_count: int | None = None,
    open_bug_count: int | None = None,
) -> bool:
    """In verify rounds the open counts (carried plus new findings) drive the
    policy, so an unfixed bug from an earlier round still fails fail_on=bugs."""
    policy = (fail_on or "never").strip().lower()
    issue_count = len(issues) if open_issue_count is None else open_issue_count
    bug_count = (
        sum(1 for issue in issues if issue.severity == "bug")
        if open_bug_count is None
        else open_bug_count
    )
    if policy == "never":
        return False
    if policy == "bugs":
        return bug_count > 0
    if policy == "any":
        return issue_count > 0
    raise ValueError(f"fail_on must be never, bugs, or any; got {fail_on!r}")


def render_review(
    *,
    collected: CollectedReview,
    lanes: list[LaneResult],
    issues: list[MergedIssue],
    verdict: str,
    run_url: str = "",
    judge_note: str = "",
    judge_cost: float | None = None,
    judge_ran: bool = False,
    reviewed_sha: str | None = None,
    extra_notices: list[str] | None = None,
    hidden_marker: str | None = None,
    round_lines: list[str] | None = None,
) -> str:
    """Single-body rendering; the first part of render_review_parts."""
    return render_review_parts(
        collected=collected,
        lanes=lanes,
        issues=issues,
        verdict=verdict,
        run_url=run_url,
        judge_note=judge_note,
        judge_cost=judge_cost,
        judge_ran=judge_ran,
        reviewed_sha=reviewed_sha,
        extra_notices=extra_notices,
        hidden_marker=hidden_marker,
        round_lines=round_lines,
    )[0]


def render_review_parts(
    *,
    collected: CollectedReview,
    lanes: list[LaneResult],
    issues: list[MergedIssue],
    verdict: str,
    run_url: str = "",
    judge_note: str = "",
    judge_cost: float | None = None,
    judge_ran: bool = False,
    reviewed_sha: str | None = None,
    extra_notices: list[str] | None = None,
    hidden_marker: str | None = None,
    round_lines: list[str] | None = None,
) -> list[str]:
    """The review body plus continuation-comment bodies for long findings
    lists, so nothing is dropped to fit GitHub's body limit.

    `hidden_marker` (the loop ledger) sits directly under the heading so it
    can never be cut by body truncation; `round_lines` is the visible
    verify-round resolution report."""
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
            if lane.thought_signature_recoveries:
                extra += (
                    f", {lane.thought_signature_recoveries} thought-signature "
                    f"{'recovery' if lane.thought_signature_recoveries == 1 else 'recoveries'}"
                )
            if lane.provider:
                extra += f", via {lane.provider}"
            cost_frag = _lane_cost_fragment(lane)
            if cost_frag:
                extra += f", {cost_frag}"
            lane_lines.append(f"- `{lane.model}`: ok ({extra})")
        else:
            spent = _lane_cost_spent_fragment(lane)
            recovery = ""
            if lane.thought_signature_recoveries:
                recovery = (
                    f"; {lane.thought_signature_recoveries} thought-signature "
                    f"{'recovery' if lane.thought_signature_recoveries == 1 else 'recoveries'}"
                )
            lane_lines.append(
                f"- `{lane.model}`: failed-open{spent}{recovery} — "
                f"{neutralize_mentions(lane.error or 'unknown error')}"
            )

    header = ["## OpenRouter pull-request review"]
    if hidden_marker:
        header.append(hidden_marker)
    header += [
        "",
        f"**Verdict:** `{verdict}`",
        f"**Scope:** `{collected.plan.scope}` ({collected.plan.kind})",
        f"**Mode:** `{collected.mode}`",
        f"**Commit:** `{reviewed_sha or collected.head_sha}`",
    ]
    if judge_note:
        header.append(f"**Judge:** {judge_note}")
    cost_note = _cost_note(lanes, judge_cost, judge_ran)
    if cost_note:
        header.append(f"**Cost:** {cost_note}")
    header.extend(
        [
            "",
            "### Lanes",
            "",
            *lane_lines,
            "",
        ]
    )
    if round_lines:
        header.extend([*_cap_block(round_lines, ROUND_REPORT_BYTES), ""])
    if collected.truncation.truncated and collected.truncation.notice:
        if collected.truncation.forces_partial:
            header.extend(
                [
                    "> This is a **partial** review. The embedded diff was truncated. "
                    "It must not be treated as a clean review.",
                    "",
                    collected.truncation.notice,
                    "",
                ]
            )
        else:
            # Diff-budget triage: every changed file was embedded or stubbed,
            # so the coverage contract still spans the whole PR — inform,
            # don't degrade the verdict.
            header.extend([f"> {collected.truncation.notice}", ""])
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


def _fmt_cost(value: float, precision: int | None = None) -> str:
    """OpenRouter credits are USD. Two decimals reads best above a dime;
    four below it so cheap lanes do not render as $0.00. Pass `precision`
    to keep every figure in one note visually consistent (so a breakdown
    adds up to its total on the page)."""
    if precision is None:
        precision = 2 if value >= 0.1 else 4
    return f"${value:.{precision}f}"


def _lane_cost_fragment(lane: LaneResult) -> str | None:
    if lane.cost_usd is not None:
        return _fmt_cost(lane.cost_usd)
    if lane.known_cost_usd is not None:
        return f"at least {_fmt_cost(lane.known_cost_usd)} (incomplete)"
    return None


def _lane_cost_spent_fragment(lane: LaneResult) -> str:
    if lane.cost_usd is not None:
        return f" ({_fmt_cost(lane.cost_usd)} spent)"
    if lane.known_cost_usd is not None:
        return f" (at least {_fmt_cost(lane.known_cost_usd)} spent, incomplete)"
    return ""


def _cost_precision(*values: float) -> int:
    return 2 if min(values) >= 0.1 else 4


def _cost_note(lanes: list[LaneResult], judge_cost: float | None, judge_ran: bool = False) -> str:
    complete_lanes = [lane for lane in lanes if lane.cost_usd is not None]
    partial_lanes = [
        lane for lane in lanes if lane.cost_usd is None and lane.known_cost_usd is not None
    ]
    unknown_lanes = [
        lane for lane in lanes if lane.cost_usd is None and lane.known_cost_usd is None
    ]
    complete_lane_sum = sum(lane.cost_usd for lane in complete_lanes)
    partial_sum = sum(lane.known_cost_usd for lane in partial_lanes)
    judge_complete = judge_cost is not None
    judge_unknown = judge_ran and judge_cost is None

    if not complete_lanes and not partial_lanes and not judge_complete:
        if judge_unknown:
            unreported = [f"`{lane.model}`" for lane in lanes]
            if unreported:
                return (
                    "unavailable — incomplete: no cost reported for "
                    f"{', '.join(unreported)}, the judge"
                )
            return "unavailable — incomplete: judge cost unreported"
        return ""

    fully_complete = not partial_lanes and not unknown_lanes and not judge_unknown
    if fully_complete:
        total = complete_lane_sum + (judge_cost or 0.0)
        figures = [total, *[lane.cost_usd for lane in complete_lanes]]
        if judge_cost is not None:
            figures.append(judge_cost)
        precision = _cost_precision(*figures)
        note = _fmt_cost(total, precision)
        if judge_cost is not None and complete_lanes:
            note += (
                f" (lanes {_fmt_cost(complete_lane_sum, precision)}"
                f" + judge {_fmt_cost(judge_cost, precision)})"
            )
        return note

    parts: list[str] = []
    if complete_lane_sum or judge_complete:
        subtotal = complete_lane_sum + (judge_cost or 0.0)
        figures = [subtotal]
        if complete_lane_sum:
            figures.append(complete_lane_sum)
        if judge_complete:
            figures.append(judge_cost)
        precision = _cost_precision(*figures)
        if judge_complete and complete_lanes:
            parts.append(
                f"{_fmt_cost(subtotal, precision)} "
                f"(lanes {_fmt_cost(complete_lane_sum, precision)}"
                f" + judge {_fmt_cost(judge_cost, precision)})"
            )
        elif judge_complete:
            parts.append(_fmt_cost(judge_cost, precision))
        else:
            parts.append(_fmt_cost(complete_lane_sum, precision))
    if partial_sum:
        precision = _cost_precision(partial_sum)
        parts.append(f"at least {_fmt_cost(partial_sum, precision)} (incomplete)")

    note = " + ".join(parts) if parts else "unavailable"
    unreported = [f"`{lane.model}`" for lane in unknown_lanes]
    if judge_unknown:
        unreported.append("the judge")
    if unreported:
        note += f" — incomplete: no cost reported for {', '.join(unreported)}"
    return note


def _cap_block(lines: list[str], max_bytes: int) -> list[str]:
    kept: list[str] = []
    used = 0
    for index, line in enumerate(lines):
        cost = len(line.encode("utf-8")) + 1
        if used + cost > max_bytes:
            omitted = len(lines) - index
            kept.append(
                f"- [{omitted} more resolution line(s) omitted; the full state "
                "is carried in the ledger]"
            )
            break
        kept.append(line)
        used += cost
    return kept


def _finalize(lines: list[str]) -> str:
    text = "\n".join(lines).rstrip() + "\n"
    if len(text.encode("utf-8")) > MAX_REVIEW_BYTES:
        clipped = text.encode("utf-8")[: MAX_REVIEW_BYTES - 80].decode("utf-8", errors="ignore")
        text = clipped.rstrip() + "\n\n[review body truncated]\n"
    return text


def inline_review_comments(
    issues: list[MergedIssue],
    *,
    allowed_lines: dict[str, set[int]],
    generation: str,
) -> list[dict[str, Any]]:
    """Inline review comments for findings that anchor inside a diff hunk.

    GitHub rejects the ENTIRE batched review if any one comment's line is not
    part of the diff, so anchors are validated against the new-side hunk
    lines. Out-of-hunk and out-of-diff (blast-radius) findings stay body-only.
    """
    comments: list[dict[str, Any]] = []
    for issue in issues:
        if not issue.file or issue.line is None:
            continue
        if issue.line not in allowed_lines.get(issue.file, set()):
            continue
        marker = f"{finding_marker(issue.id, generation)}\n" if issue.id and generation else ""
        comments.append(
            {
                "path": issue.file,
                "line": issue.line,
                "side": "RIGHT",
                "body": (
                    f"{marker}{severity_emoji(issue.severity)} "
                    f"**{neutralize_mentions(issue.title)}** (`{issue.severity}`)\n\n"
                    f"{neutralize_mentions(issue.body)}"
                ),
            }
        )
    return comments


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
