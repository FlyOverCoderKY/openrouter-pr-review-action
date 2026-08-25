"""Deterministic judge: merge, de-dupe, and format attribution lines."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from or_pr_review.schema import Finding, LaneResult

_WORD_RE = re.compile(r"[a-z0-9]+")
_SEVERITY_RANK = {"bug": 2, "risk": 1, "nit": 0}


def neutralize_mentions(value: str) -> str:
    """Stop model-authored text from pinging GitHub users or teams."""
    return value.replace("@", "@\u200b")


@dataclass
class MergedIssue:
    title: str
    body: str
    severity: str
    file: str | None
    line: int | None
    models: list[str] = field(default_factory=list)
    id: str | None = None  # ledger finding id (r<round>-<n>), assigned at finish

    def heading(self, number: int) -> str:
        return f"Issue {number} - {neutralize_mentions(self.title)} ({identified_by(self.models)})"


def identified_by(models: list[str]) -> str:
    names = _unique(models)
    if not names:
        return "identified by unknown"
    if len(names) == 1:
        return f"identified by {names[0]}"
    if len(names) == 2:
        return f"identified by {names[0]} and {names[1]}"
    return f"identified by {', '.join(names[:-1])}, and {names[-1]}"


def format_issue_block(number: int, issue: MergedIssue) -> str:
    heading = issue.heading(number)
    parts = [heading, ""]
    location = _location_line(issue)
    if location:
        parts.append(location)
    parts.append(f"Severity: {issue.severity}")
    parts.append("")
    parts.append(neutralize_mentions(issue.body.rstrip()))
    return "\n".join(parts).rstrip() + "\n"


def issues_from_single_lane(lane: LaneResult) -> list[MergedIssue]:
    """Post one lane as-is. No merge, no de-dupe, no cross-model attribution."""
    issues = [
        MergedIssue(
            title=finding.title,
            body=finding.body,
            severity=finding.severity,
            file=finding.file,
            line=finding.line,
            models=[finding.model_id or lane.model],
        )
        for finding in lane.findings
    ]
    issues.sort(key=_sort_key)
    return issues


def merge_lanes(lanes: list[LaneResult]) -> list[MergedIssue]:
    """Local de-dupe used by tests and as a last-resort helper. The multi-lane
    product path calls the OpenRouter judge instead.
    """
    merged: list[MergedIssue] = []
    for lane in lanes:
        if not lane.ok:
            continue
        for finding in lane.findings:
            _absorb(merged, finding)
    merged.sort(key=_sort_key)
    return merged


def _absorb(merged: list[MergedIssue], finding: Finding) -> None:
    for existing in merged:
        if _same_issue(existing, finding):
            if finding.model_id not in existing.models:
                existing.models.append(finding.model_id)
            if _SEVERITY_RANK.get(finding.severity, 0) > _SEVERITY_RANK.get(existing.severity, 0):
                existing.severity = finding.severity
            if len(finding.body) > len(existing.body):
                existing.body = finding.body
            if existing.line is None and finding.line is not None:
                existing.line = finding.line
            if existing.file is None and finding.file:
                existing.file = finding.file
            return
    merged.append(
        MergedIssue(
            title=finding.title,
            body=finding.body,
            severity=finding.severity,
            file=finding.file,
            line=finding.line,
            models=[finding.model_id],
        )
    )


def _same_issue(existing: MergedIssue, finding: Finding) -> bool:
    title_a = _normalize(existing.title)
    title_b = _normalize(finding.title)
    file_a = (existing.file or "").strip()
    file_b = (finding.file or "").strip()
    if title_a == title_b and file_a == file_b:
        return True
    if file_a and file_b and file_a == file_b and _jaccard(_tokens(existing.title), _tokens(finding.title)) >= 0.7:
        return True
    if title_a == title_b and (not file_a or not file_b):
        return True
    return False


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _location_line(issue: MergedIssue) -> str | None:
    if issue.file and issue.line:
        return f"Location: `{issue.file}:{issue.line}`"
    if issue.file:
        return f"Location: `{issue.file}`"
    return None


def _sort_key(issue: MergedIssue) -> tuple[int, str, str]:
    return (-_SEVERITY_RANK.get(issue.severity, 0), issue.file or "", issue.title.lower())
