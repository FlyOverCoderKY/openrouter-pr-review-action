"""Deterministic judge: merge, de-dupe, and format attribution lines."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from or_pr_review.schema import SEVERITY_RANK, Finding, LaneResult

_WORD_RE = re.compile(r"[a-z0-9]+")
_REVIEW_ENVIRONMENT_RE = re.compile(
    r"\b(?:review|provided|temporary|inert)\s+"
    r"(?:checkout|workspace|environment|snapshot)\b"
    r"|\b(?:checkout|workspace|snapshot)\s+(?:provided|supplied)\s+for\s+"
    r"(?:this\s+)?review\b"
    r"|\breview tooling\b|\btool access\b|\brepository snapshot\b",
    re.IGNORECASE,
)
_ACCESS_FAILURE_RE = re.compile(
    r"\b(?:can(?:not|'t)|could(?: not|n't)|unable to|failed to)\s+"
    r"(?:be\s+)?(?:open(?:ed)?|read|search(?:ed)?|access(?:ed)?|inspect(?:ed)?|"
    r"locate(?:d)?|find|load(?:ed)?|verify|stat)\b"
    r"|\b(?:not present(?:/readable)?|not readable|unavailable|inaccessible)\b",
    re.IGNORECASE,
)
_DUPLICATE_NOISE_TOKENS = {
    "a",
    "also",
    "an",
    "both",
    "largely",
    "new",
    "that",
    "the",
    "these",
    "this",
    "those",
}


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


def identified_by(models: list[str]) -> str:
    names = _unique(models)
    if not names:
        return "identified by unknown"
    if len(names) == 1:
        return f"identified by {names[0]}"
    if len(names) == 2:
        return f"identified by {names[0]} and {names[1]}"
    return f"identified by {', '.join(names[:-1])}, and {names[-1]}"


SEVERITY_EMOJI = {"bug": "\U0001f534", "risk": "\U0001f7e0", "nit": "\U0001f535"}


def severity_emoji(severity: str) -> str:
    return SEVERITY_EMOJI.get(severity, "⚪")


def format_issue_block(number: int, issue: MergedIssue) -> str:
    """A scannable block: emoji-severity heading, one metadata line, body."""
    heading = (
        f"#### {severity_emoji(issue.severity)} Issue {number} — {neutralize_mentions(issue.title)}"
    )
    meta = [f"`{issue.severity}`"]
    if issue.file and issue.line:
        meta.insert(0, f"`{issue.file}:{issue.line}`")
    elif issue.file:
        meta.insert(0, f"`{issue.file}`")
    meta.append(identified_by(issue.models))
    parts = [heading, "", " · ".join(meta), "", neutralize_mentions(issue.body.rstrip())]
    return "\n".join(parts).rstrip() + "\n"


def merged_issue_from_finding(finding: Finding, *, fallback_model: str = "") -> MergedIssue:
    """Convert the validated lane domain object to a publishable issue.

    Keeping this boundary here makes the local fallback and judge repairs use
    exactly the same Finding-to-MergedIssue representation and attribution.
    """
    model = finding.model_id or fallback_model
    return MergedIssue(
        title=finding.title,
        body=finding.body,
        severity=finding.severity,
        file=finding.file,
        line=finding.line,
        models=[model] if model else [],
    )


def issues_from_single_lane(lane: LaneResult) -> list[MergedIssue]:
    """Post one lane as-is, except for review-environment diagnostics.

    A model's inability to read the supplied checkout is useful operational
    telemetry, but it is not evidence of a defect in the pull request. Keep
    it in the action log instead of turning it into a GitHub finding.
    """
    issues = []
    for finding in lane.findings:
        if is_environmental_diagnostic(
            title=finding.title,
            body=finding.body,
            line=finding.line,
        ):
            print(
                f"lane diagnostic from {finding.model_id or lane.model}: "
                f"{finding.title} (not published as a code finding)"
            )
            continue
        issues.append(merged_issue_from_finding(finding, fallback_model=lane.model))
    issues.sort(key=_sort_key)
    return issues


def is_environmental_diagnostic(*, title: str, body: str, line: int | None) -> bool:
    """Return whether text reports a reviewer/tool limitation, not a code bug.

    The classifier is intentionally narrow: the finding must be unanchored,
    explicitly name the temporary or supplied review environment, and report
    an access/read failure. A real missing-file or runtime-access finding that
    merely mentions the application's current environment or workspace remains
    publishable.
    """
    if line is not None:
        return False
    text = f"{title}\n{body}"
    return bool(_REVIEW_ENVIRONMENT_RE.search(text) and _ACCESS_FAILURE_RE.search(text))


def same_merged_issue(
    left: MergedIssue,
    right: MergedIssue,
    *,
    title_similarity: float = 0.8,
    require_title_noise_only: bool = True,
    require_evidence_agreement: bool = True,
) -> bool:
    """Conservative exact/near duplicate test for already-merged issues.

    Near matches require the same anchored location plus strong agreement in
    both title and evidence text. This catches clerical judge duplication
    without folding two different defects that merely live on the same line.
    """
    file_left = (left.file or "").strip()
    file_right = (right.file or "").strip()
    if file_left != file_right or left.line != right.line:
        return False

    title_left = _normalize(left.title)
    title_right = _normalize(right.title)
    if title_left == title_right:
        return not require_evidence_agreement or _evidence_agrees(left.body, right.body)
    if not file_left or left.line is None:
        return False
    if _jaccard(_tokens(left.title), _tokens(right.title)) < title_similarity:
        return False
    if require_title_noise_only and (
        (_tokens(left.title) ^ _tokens(right.title)) - _DUPLICATE_NOISE_TOKENS
    ):
        return False
    return not require_evidence_agreement or _evidence_agrees(left.body, right.body)


def _evidence_agrees(left: str, right: str) -> bool:
    """Strong, deterministic evidence agreement for duplicate suppression.

    Exact normalized bodies are safe. The only non-exact allowance is a high
    token overlap whose differences are all low-information articles or
    qualifiers. Substantive token differences preserve both findings.
    """
    if _normalize(left) == _normalize(right):
        return True
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if _jaccard(left_tokens, right_tokens) < 0.85:
        return False
    return not ((left_tokens ^ right_tokens) - _DUPLICATE_NOISE_TOKENS)


def absorb_merged_issue(existing: MergedIssue, incoming: MergedIssue) -> None:
    """Combine duplicate metadata while retaining the strongest evidence."""
    for model in incoming.models:
        if model and model not in existing.models:
            existing.models.append(model)
    if SEVERITY_RANK.get(incoming.severity, 0) > SEVERITY_RANK.get(existing.severity, 0):
        existing.severity = incoming.severity
    if len(incoming.body) > len(existing.body):
        existing.body = incoming.body
    if existing.line is None and incoming.line is not None:
        existing.line = incoming.line
    if existing.file is None and incoming.file:
        existing.file = incoming.file


def deduplicate_issues(
    issues: list[MergedIssue],
    *,
    title_similarity: float = 0.8,
    require_title_noise_only: bool = True,
    require_evidence_agreement: bool = True,
) -> tuple[list[MergedIssue], int]:
    """Deterministically suppress exact/conservative-near duplicate issues.

    Returns fresh issue objects and the number of duplicate rows absorbed so
    callers can surface judge repairs in diagnostics.
    """
    merged: list[MergedIssue] = []
    absorbed = 0
    for issue in issues:
        candidate = MergedIssue(
            title=issue.title,
            body=issue.body,
            severity=issue.severity,
            file=issue.file,
            line=issue.line,
            models=list(issue.models),
            id=issue.id,
        )
        for existing in merged:
            if same_merged_issue(
                existing,
                candidate,
                title_similarity=title_similarity,
                require_title_noise_only=require_title_noise_only,
                require_evidence_agreement=require_evidence_agreement,
            ):
                absorb_merged_issue(existing, candidate)
                absorbed += 1
                break
        else:
            merged.append(candidate)
    return merged, absorbed


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


def _sort_key(issue: MergedIssue) -> tuple[int, str, str]:
    return (-SEVERITY_RANK.get(issue.severity, 0), issue.file or "", issue.title.lower())
