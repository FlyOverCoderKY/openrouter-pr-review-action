"""Review-loop state: a carried-findings ledger embedded invisibly in the
bot's own posted review bodies, with bounded evidence per finding.

The ledger is bot-internal memory, not a contract with the fixing agent:
agents respond through ordinary commits and comment-thread replies, and the
model adjudicates dispositions from those signals each round. Only review
bodies authored by the configured bot login are trusted. The newest marker
is authoritative: if it fails to decode, state recovery fails closed rather
than silently resetting the loop or rolling back to an older round.

Unlike the sibling Grok ledger, each carried finding keeps a bounded
`evidence` excerpt of its original body, so verify rounds adjudicate with
the failure scenario in view rather than an 80-character title.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass, replace

from or_pr_review.errors import ActionError
from or_pr_review.merge import MergedIssue, neutralize_mentions
from or_pr_review.schema import SEVERITIES, Resolution, valid_review_path

LEDGER_VERSION = 1
LEDGER_PREFIX = "<!-- openrouter-review-ledger:v1:"
LEDGER_SUFFIX = " -->"
MAX_LEDGER_FINDINGS = 200
MAX_LEDGER_BYTES = 40_000
MAX_ROUNDS_TRACKED = 999
MAX_EVIDENCE = 600
TRIMMED_EVIDENCE = 200
MAX_LEDGER_TITLE = 160
TRIMMED_TITLE = 100
MAX_LEDGER_MODELS = 8
MAX_REPLY_CHARS = 2_000
MAX_REPLIES_BYTES = 16_000

_FINDING_ID_RE = re.compile(r"^r\d{1,3}-\d{1,3}$")
_LEDGER_RE = re.compile(
    re.escape(LEDGER_PREFIX) + r"([A-Za-z0-9+/=]+)" + re.escape(LEDGER_SUFFIX)
)
FINDING_MARKER_RE = re.compile(r"<!-- or-finding:(r\d{1,3}-\d{1,3}) -->")

_STATUS_ICONS = {
    "fixed": "✅",
    "not_fixed": "❌",
    "fixed_incorrectly": "⚠️",
    "disputed": "🤝",
    "unaddressed": "⏳",
}
_SEVERITY_RANK = {"bug": 2, "risk": 1, "nit": 0}
# Conservative cross-lane merge: a higher rank always wins, so a finding is
# `fixed` only when every lane that resolved it says fixed, and any dispute
# settles it.
_RESOLUTION_RANK = {"fixed": 0, "not_fixed": 1, "fixed_incorrectly": 2, "disputed": 3}


@dataclass(frozen=True)
class LedgerFinding:
    id: str
    severity: str  # nit | risk | bug
    file: str | None
    line: int | None
    title: str
    evidence: str
    status: str  # open | disputed
    models: tuple[str, ...] = ()


@dataclass(frozen=True)
class Ledger:
    round_number: int
    findings: tuple[LedgerFinding, ...]
    reviewed_sha: str = ""


@dataclass(frozen=True)
class LoopState:
    """The loop position of the current run."""

    mode: str  # initial | verify
    round_number: int
    prior_findings: tuple[LedgerFinding, ...] = ()

    @property
    def open_prior(self) -> tuple[LedgerFinding, ...]:
        return tuple(f for f in self.prior_findings if f.status == "open")

    @property
    def disputed_prior(self) -> tuple[LedgerFinding, ...]:
        return tuple(f for f in self.prior_findings if f.status == "disputed")


@dataclass(frozen=True)
class RoundOutcome:
    ledger: Ledger
    issues: list[MergedIssue]
    resolution_lines: list[str]
    open_issue_count: int
    open_bug_count: int


def decide_loop_state(
    *, review_mode: str, event_action: str, ledger: Ledger | None
) -> tuple[str, int]:
    """Return (mode, round_number). An initial review always resets the loop."""
    if review_mode == "initial" or ledger is None:
        return "initial", 1
    next_round = min(ledger.round_number + 1, MAX_ROUNDS_TRACKED)
    if review_mode == "verify":
        return "verify", next_round
    if (event_action or "").strip().lower() == "synchronize":
        return "verify", next_round
    return "initial", 1


def finding_marker(finding_id: str) -> str:
    """Invisible marker tying an inline comment to a ledger finding id."""
    return f"<!-- or-finding:{finding_id} -->"


def merge_resolutions(
    per_lane: list[list[Resolution]], prior_ids: set[str]
) -> dict[str, Resolution]:
    """Deterministic conservative merge of lane resolutions (no LLM involved)."""
    merged: dict[str, Resolution] = {}
    for resolutions in per_lane:
        for resolution in resolutions:
            if resolution.id not in prior_ids:
                continue
            current = merged.get(resolution.id)
            if current is None or (
                _RESOLUTION_RANK[resolution.status] > _RESOLUTION_RANK[current.status]
            ):
                merged[resolution.id] = resolution
    return merged


def apply_round(
    state: LoopState,
    issues: list[MergedIssue],
    resolutions: dict[str, Resolution],
) -> RoundOutcome:
    """Fold a completed round into the ledger and number the new issues."""
    carried: list[LedgerFinding] = []
    resolution_lines: list[str] = []
    for finding in state.open_prior:
        resolution = resolutions.get(finding.id)
        status = resolution.status if resolution else "unaddressed"
        note = resolution.note if resolution else ""
        resolution_lines.append(_resolution_line(finding, status, note))
        if status == "fixed":
            continue
        if status == "disputed":
            carried.append(replace(finding, status="disputed"))
            continue
        carried.append(finding)
    carried.extend(state.disputed_prior)

    numbered: list[MergedIssue] = []
    for issue in issues:
        numbered.append(replace(issue, id=f"r{state.round_number}-{len(numbered) + 1}"))
    carried.extend(
        LedgerFinding(
            id=issue.id or "",
            severity=issue.severity,
            file=issue.file,
            line=issue.line,
            title=issue.title[:MAX_LEDGER_TITLE],
            evidence=issue.body[:MAX_EVIDENCE],
            status="open",
            models=tuple(issue.models[:MAX_LEDGER_MODELS]),
        )
        for issue in numbered
    )

    ledger = Ledger(round_number=state.round_number, findings=tuple(carried))
    open_findings = [f for f in ledger.findings if f.status == "open"]
    open_bug_count = sum(1 for f in open_findings if f.severity == "bug")
    return RoundOutcome(
        ledger=ledger,
        issues=numbered,
        resolution_lines=resolution_lines,
        open_issue_count=len(open_findings),
        open_bug_count=open_bug_count,
    )


def round_report(state: LoopState, outcome: RoundOutcome) -> list[str]:
    """The visible verify-round resolution section for the review body."""
    if state.mode != "verify":
        return []
    lines = [f"### Round {state.round_number} resolution", ""]
    lines.extend(outcome.resolution_lines or ["- No prior findings were open."])
    lines.append(
        f"- Open findings after this round: {outcome.open_issue_count} "
        f"({outcome.open_bug_count} bug-severity)."
    )
    return lines


def encode_ledger(ledger: Ledger, *, repo: str, pr_number: int) -> str:
    """Encode a marker bound to this repo/PR that is guaranteed to decode.

    Findings are kept in priority order (open bugs, open risks, open nits,
    then disputed). To fit the byte budget the encoder first drops settled
    disputed findings, then shrinks evidence and titles. Losing an open
    finding is worse than failing visibly, so if the open findings alone
    cannot be persisted this raises instead of silently discarding state.
    """
    ordered = _trim_order(list(ledger.findings))
    open_count = sum(1 for finding in ordered if finding.status == "open")
    if open_count > MAX_LEDGER_FINDINGS:
        raise ActionError(_overflow_message(open_count))
    findings = ordered[:MAX_LEDGER_FINDINGS]
    for evidence_cap, title_cap in (
        (MAX_EVIDENCE, MAX_LEDGER_TITLE),
        (TRIMMED_EVIDENCE, TRIMMED_TITLE),
        (0, TRIMMED_TITLE),
    ):
        candidate = [
            replace(
                finding,
                evidence=finding.evidence[:evidence_cap],
                title=finding.title[:title_cap].strip() or "…",
            )
            for finding in findings
        ]
        while True:
            encoded = _encode(ledger, candidate, repo=repo, pr_number=pr_number)
            if len(encoded) <= MAX_LEDGER_BYTES:
                token = encoded[len(LEDGER_PREFIX) : -len(LEDGER_SUFFIX)]
                if _decode(token, repo=repo, pr_number=pr_number) is None:
                    raise ActionError(
                        "the review-loop ledger failed its decode self-check; "
                        "refusing to publish invalid state"
                    )
                return encoded
            if candidate and candidate[-1].status == "disputed":
                candidate = candidate[:-1]
                continue
            break
    raise ActionError(_overflow_message(open_count))


def _overflow_message(open_count: int) -> str:
    return (
        f"review-loop state overflow: {open_count} open findings cannot all be "
        "persisted in the ledger; fix or dispute findings to shrink the backlog, "
        "or reset the loop with review_mode: initial and review_scope: full-pr"
    )


def _trim_order(findings: list[LedgerFinding]) -> list[LedgerFinding]:
    def priority(finding: LedgerFinding) -> tuple[int, int]:
        status_rank = 0 if finding.status == "open" else 1
        return (status_rank, -_SEVERITY_RANK.get(finding.severity, 0))

    return sorted(findings, key=priority)


def _encode(
    ledger: Ledger, findings: list[LedgerFinding], *, repo: str, pr_number: int
) -> str:
    payload = {
        "lv": LEDGER_VERSION,
        "repo": repo,
        "pr": pr_number,
        "sha": ledger.reviewed_sha,
        "round": ledger.round_number,
        "findings": [
            {
                "id": finding.id,
                "sev": finding.severity,
                "file": finding.file,
                "line": finding.line,
                "title": finding.title,
                "ev": finding.evidence,
                "st": finding.status,
                "m": list(finding.models),
            }
            for finding in findings
        ],
    }
    compact = json.dumps(payload, separators=(",", ":"))
    token = base64.b64encode(compact.encode("utf-8")).decode("ascii")
    return f"{LEDGER_PREFIX}{token}{LEDGER_SUFFIX}"


def has_ledger_marker(body: str) -> bool:
    return LEDGER_PREFIX in body


def extract_ledger(body: str, *, repo: str, pr_number: int) -> Ledger | None:
    """Decode the newest valid ledger marker bound to this repo/PR, else None."""
    for match in reversed(list(_LEDGER_RE.finditer(body))):
        ledger = _decode(match.group(1), repo=repo, pr_number=pr_number)
        if ledger is not None:
            return ledger
    return None


def latest_ledger(bodies: list[str], *, repo: str, pr_number: int) -> Ledger | None:
    """The ledger from the newest marker-bearing body, or None when state-free.

    The newest marker is authoritative: if it fails to decode, this raises
    rather than falling back to an older valid ledger, because rolling state
    back a round would silently resurrect resolved findings and lose new ones.
    """
    for body in reversed(bodies):
        if not has_ledger_marker(body):
            continue
        ledger = extract_ledger(body, repo=repo, pr_number=pr_number)
        if ledger is None:
            raise ActionError(
                "the newest review-loop ledger on this PR is corrupted, from a "
                "different harness version, or bound to a different repository; "
                "force review_mode: initial with review_scope: full-pr to reset it"
            )
        return ledger
    return None


def _decode(token: str, *, repo: str, pr_number: int) -> Ledger | None:
    try:
        payload = json.loads(base64.b64decode(token, validate=True).decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("lv") != LEDGER_VERSION:
        return None
    if payload.get("repo") != repo or payload.get("pr") != pr_number:
        return None
    sha = payload.get("sha")
    if not isinstance(sha, str) or (sha != "" and not re.fullmatch(r"[0-9a-f]{40}", sha)):
        return None
    round_number = payload.get("round")
    raw_findings = payload.get("findings")
    if (
        isinstance(round_number, bool)
        or not isinstance(round_number, int)
        or not 1 <= round_number <= MAX_ROUNDS_TRACKED
        or not isinstance(raw_findings, list)
        or len(raw_findings) > MAX_LEDGER_FINDINGS
    ):
        return None
    findings: list[LedgerFinding] = []
    for item in raw_findings:
        finding = _decode_finding(item)
        if finding is None:
            return None
        findings.append(finding)
    return Ledger(round_number=round_number, findings=tuple(findings), reviewed_sha=sha)


def _decode_finding(item: object) -> LedgerFinding | None:
    if not isinstance(item, dict):
        return None
    ident = item.get("id")
    severity = item.get("sev")
    file_value = item.get("file")
    line = item.get("line")
    title = item.get("title")
    evidence = item.get("ev")
    status = item.get("st")
    models = item.get("m")
    if not isinstance(ident, str) or not _FINDING_ID_RE.fullmatch(ident):
        return None
    if severity not in SEVERITIES:
        return None
    if file_value is not None and (
        not isinstance(file_value, str) or not valid_review_path(file_value)
    ):
        return None
    if line is not None and (isinstance(line, bool) or not isinstance(line, int) or line <= 0):
        return None
    if not isinstance(title, str) or not title.strip() or len(title) > 300:
        return None
    if not isinstance(evidence, str) or len(evidence) > MAX_EVIDENCE + 16:
        return None
    if status not in {"open", "disputed"}:
        return None
    if not isinstance(models, list) or len(models) > MAX_LEDGER_MODELS:
        return None
    if not all(isinstance(model, str) and len(model) <= 100 for model in models):
        return None
    return LedgerFinding(
        id=ident,
        severity=severity,
        file=file_value,
        line=line,
        title=title.strip(),
        evidence=evidence,
        status=status,
        models=tuple(models),
    )


def render_agent_context(
    finding_replies: list[tuple[str, str, str]],
    issue_comments: list[tuple[str, str]],
) -> str:
    """Render comment-thread replies and PR comments into a bounded prompt block."""
    lines: list[str] = []
    for finding_id, login, body in finding_replies:
        lines.append(f"Reply to finding {finding_id} (from {login}):")
        lines.append(_clip_reply(body))
        lines.append("")
    for login, body in issue_comments:
        lines.append(f"PR comment (from {login}):")
        lines.append(_clip_reply(body))
        lines.append("")
    text = "\n".join(lines).strip()
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_REPLIES_BYTES:
        text = encoded[:MAX_REPLIES_BYTES].decode("utf-8", errors="ignore")
        text += "\n…[additional replies omitted]"
    return text


def _clip_reply(body: str) -> str:
    text = body.strip()
    if len(text) > MAX_REPLY_CHARS:
        return text[:MAX_REPLY_CHARS] + "…[clipped]"
    return text


def _resolution_line(finding: LedgerFinding, status: str, note: str) -> str:
    icon = _STATUS_ICONS.get(status, "⏳")
    label = status.replace("_", " ")
    text = f"- {icon} `{finding.id}` {label} — **{neutralize_mentions(finding.title)}**"
    if note:
        text += f": {neutralize_mentions(note)}"
    return text
