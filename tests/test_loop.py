from __future__ import annotations

import base64

import pytest

from or_pr_review.errors import ActionError
from or_pr_review.loop import (
    LEDGER_PREFIX,
    Ledger,
    LedgerFinding,
    LoopState,
    apply_round,
    decide_loop_state,
    encode_ledger,
    extract_ledger,
    latest_ledger,
    merge_resolutions,
    render_agent_context,
    round_report,
)
from or_pr_review.merge import MergedIssue
from or_pr_review.schema import Resolution

REPO = "o/r"


def _finding(
    ident: str = "r1-1",
    status: str = "open",
    severity: str = "bug",
    evidence: str = "the failure detail",
) -> LedgerFinding:
    return LedgerFinding(
        id=ident,
        severity=severity,
        file="src/app.py",
        line=3,
        title="Missing auth check",
        evidence=evidence,
        status=status,
        models=("x-ai/grok-4.6",),
    )


def test_ledger_roundtrip_and_binding() -> None:
    ledger = Ledger(round_number=2, findings=(_finding(),), reviewed_sha="a" * 40)
    marker = encode_ledger(ledger, repo=REPO, pr_number=7)
    assert marker.startswith(LEDGER_PREFIX)
    decoded = extract_ledger(marker, repo=REPO, pr_number=7)
    assert decoded is not None
    assert decoded.round_number == 2
    assert decoded.reviewed_sha == "a" * 40
    assert decoded.findings[0].evidence == "the failure detail"
    assert decoded.findings[0].models == ("x-ai/grok-4.6",)
    assert extract_ledger(marker, repo="other/repo", pr_number=7) is None
    assert extract_ledger(marker, repo=REPO, pr_number=8) is None


def test_latest_ledger_newest_marker_is_authoritative() -> None:
    valid = encode_ledger(
        Ledger(round_number=1, findings=(_finding(),), reviewed_sha="a" * 40),
        repo=REPO,
        pr_number=7,
    )
    junk = LEDGER_PREFIX + base64.b64encode(b'{"nope": 1}').decode("ascii") + " -->"
    assert latest_ledger([valid], repo=REPO, pr_number=7) is not None
    with pytest.raises(ActionError, match="reset"):
        latest_ledger([valid, junk], repo=REPO, pr_number=7)
    assert latest_ledger(["no marker here"], repo=REPO, pr_number=7) is None
    assert latest_ledger([], repo=REPO, pr_number=7) is None


def test_decide_loop_state_matrix() -> None:
    ledger = Ledger(round_number=3, findings=(), reviewed_sha="")
    assert decide_loop_state(
        review_mode="initial", event_action="synchronize", ledger=ledger
    ) == ("initial", 1)
    assert decide_loop_state(review_mode="auto", event_action="synchronize", ledger=None) == (
        "initial",
        1,
    )
    assert decide_loop_state(
        review_mode="auto", event_action="synchronize", ledger=ledger
    ) == ("verify", 4)
    assert decide_loop_state(review_mode="verify", event_action="", ledger=ledger) == (
        "verify",
        4,
    )
    assert decide_loop_state(review_mode="auto", event_action="opened", ledger=ledger) == (
        "initial",
        1,
    )


def test_apply_round_folds_resolutions_and_numbers_new_issues() -> None:
    state = LoopState(
        mode="verify",
        round_number=2,
        prior_findings=(
            _finding("r1-1"),
            _finding("r1-2", severity="risk"),
            _finding("r1-3", status="disputed"),
        ),
    )
    new_issue = MergedIssue("New race", "check-then-act", "risk", "db.py", 9, ["x-ai/grok-4.6"])
    resolutions = {"r1-1": Resolution(id="r1-1", status="fixed", note="added check")}
    outcome = apply_round(state, [new_issue], resolutions)
    statuses = {finding.id: finding.status for finding in outcome.ledger.findings}
    assert "r1-1" not in statuses  # fixed → dropped
    assert statuses["r1-2"] == "open"  # unaddressed carries forward
    assert statuses["r1-3"] == "disputed"
    assert outcome.issues[0].id == "r2-1"
    assert statuses["r2-1"] == "open"
    assert outcome.open_issue_count == 2
    assert outcome.open_bug_count == 0
    lines = "\n".join(outcome.resolution_lines)
    assert "✅" in lines and "r1-1" in lines
    assert "⏳" in lines and "r1-2" in lines
    report = round_report(state, outcome)
    assert report[0] == "### Round 2 resolution"
    assert round_report(LoopState(mode="initial", round_number=1), outcome) == []


def test_disputed_resolution_settles_a_finding() -> None:
    state = LoopState(mode="verify", round_number=3, prior_findings=(_finding("r1-1"),))
    resolutions = {"r1-1": Resolution(id="r1-1", status="disputed", note="by design")}
    outcome = apply_round(state, [], resolutions)
    assert outcome.ledger.findings[0].status == "disputed"
    assert outcome.open_issue_count == 0
    assert "🤝" in outcome.resolution_lines[0]


def test_merge_resolutions_is_conservative() -> None:
    prior = {"r1-1"}
    fixed = Resolution(id="r1-1", status="fixed", note="")
    not_fixed = Resolution(id="r1-1", status="not_fixed", note="still broken")
    disputed = Resolution(id="r1-1", status="disputed", note="by design")
    assert merge_resolutions([[fixed], [fixed]], prior)["r1-1"].status == "fixed"
    assert merge_resolutions([[fixed], [not_fixed]], prior)["r1-1"].status == "not_fixed"
    assert merge_resolutions([[not_fixed], [disputed]], prior)["r1-1"].status == "disputed"
    assert merge_resolutions([[Resolution(id="zz", status="fixed", note="")]], prior) == {}


def test_encode_ledger_trims_to_fit_and_keeps_open_findings() -> None:
    open_findings = tuple(
        LedgerFinding(
            id=f"r1-{n}",
            severity="bug",
            file=None,
            line=None,
            title="T" * 160,
            evidence="E" * 600,
            status="open",
            models=(),
        )
        for n in range(1, 60)
    )
    disputed = tuple(
        LedgerFinding(
            id=f"r2-{n}",
            severity="nit",
            file=None,
            line=None,
            title="D" * 160,
            evidence="E" * 600,
            status="disputed",
            models=(),
        )
        for n in range(1, 60)
    )
    marker = encode_ledger(
        Ledger(round_number=2, findings=open_findings + disputed, reviewed_sha=""),
        repo=REPO,
        pr_number=1,
    )
    assert len(marker) <= 40_000
    decoded = extract_ledger(marker, repo=REPO, pr_number=1)
    assert decoded is not None
    open_ids = {finding.id for finding in decoded.findings if finding.status == "open"}
    assert len(open_ids) == 59  # every open finding survives trimming


def test_encode_ledger_overflow_fails_visibly() -> None:
    too_many = tuple(
        LedgerFinding(
            id=f"r1-{n}",
            severity="nit",
            file=None,
            line=None,
            title="t",
            evidence="",
            status="open",
            models=(),
        )
        for n in range(1, 202)
    )
    with pytest.raises(ActionError, match="overflow"):
        encode_ledger(
            Ledger(round_number=1, findings=too_many, reviewed_sha=""),
            repo=REPO,
            pr_number=1,
        )


def test_render_agent_context_clips_long_replies() -> None:
    text = render_agent_context([("r1-1", "dev", "x" * 3000)], [("dev", "looks good")])
    assert "Reply to finding r1-1 (from dev):" in text
    assert "…[clipped]" in text
    assert "PR comment (from dev):" in text


def test_render_agent_context_overflow_keeps_newest() -> None:
    replies = [
        ("r1-1", "dev", "OLD reply " + "x" * 3000),
        ("r1-2", "dev", "NEWEST critical reply"),
    ]
    comments = [("dev", f"comment {n} " + "y" * 1900) for n in range(20)]
    text = render_agent_context(replies, comments)
    # Finding replies get the budget first; issue-comment overflow drops the
    # OLDEST entries, never the newest.
    assert "NEWEST critical reply" in text
    assert "comment 19 " in text
    assert "comment 0 " not in text
    assert "…[older entries omitted]" in text
    assert len(text.encode("utf-8")) <= 16_000 + 200


def test_ledger_generation_roundtrip() -> None:
    ledger = Ledger(
        round_number=1,
        findings=(_finding(),),
        reviewed_sha="a" * 40,
        generation="1234567890ab",
    )
    marker = encode_ledger(ledger, repo=REPO, pr_number=7)
    decoded = extract_ledger(marker, repo=REPO, pr_number=7)
    assert decoded is not None
    assert decoded.generation == "1234567890ab"
