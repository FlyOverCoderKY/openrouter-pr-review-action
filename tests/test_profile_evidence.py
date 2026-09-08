from __future__ import annotations

import json
from dataclasses import replace

import pytest

from or_pr_review.collect import CollectedReview, DiffPlan, Truncation
from or_pr_review.errors import SchemaError
from or_pr_review.loop import Ledger, LedgerFinding, encode_ledger
from or_pr_review.profile_evidence import (
    MAX_HISTORY_EVENTS,
    AcceptedRequest,
    VerifiedRunReceipt,
    canonical_receipt,
    evaluate_receipt,
    fold_requests,
    parse_receipt,
    parse_review_receipt,
    receipt_digest,
)
from or_pr_review.publish import render_review_parts

HEAD, BASE, DIGEST, REGISTRY, CONTEXT = "a" * 40, "b" * 40, "c" * 64, "d" * 64, "e" * 64
REPO = "owner/repo"
RUN_URL = f"https://github.example/{REPO}/actions/runs/42"
GENERATION = "f" * 12

GATE_KWARGS = {
    "repository": REPO,
    "pr_number": 9,
    "head_sha": HEAD,
    "policy_digest": DIGEST,
    "registry_digest": REGISTRY,
    "profile": "code",
    "minimum": "deep",
    "required_models": ("vendor/required",),
}


def payload(**changes):
    value = {
        "version": 1,
        "repository": REPO,
        "pr_number": 9,
        "head_sha": HEAD,
        "policy_base_sha": BASE,
        "policy_digest": DIGEST,
        "profile": "code",
        "level": "deep",
        "trigger": "manual",
        "registry_digest": REGISTRY,
        "context_sha256": CONTEXT,
        "required_models": ["vendor/required"],
        "successful_models": ["vendor/required"],
        "panel_status": "complete",
        "profile_satisfied": True,
        "verdict": "clean",
        "scope": "full-pr",
        "mode": "initial",
        "run_url": RUN_URL,
        "run_attempt": 1,
    }
    value.update(changes)
    return value


def artifact(**changes):
    return json.dumps(payload(**changes), separators=(",", ":"))


def collected(
    *,
    scope: str = "full-pr",
    kind: str = "full-pr",
    mode: str = "initial",
) -> CollectedReview:
    return CollectedReview(
        9,
        "Change",
        "",
        HEAD,
        "main",
        "feature",
        DiffPlan(scope, kind, BASE if kind != "full-pr" else None, HEAD, None),
        Truncation("", False, 0, 0, 300),
        mode,
    )


def render_body(
    receipt_dict: dict,
    review_collected: CollectedReview,
    *,
    verdict: str | None = None,
    run_url: str | None = None,
    ledger: bool = True,
) -> str:
    verdict = verdict or receipt_dict["verdict"]
    hidden_marker = None
    if ledger and verdict in {"clean", "issues"}:
        hidden_marker = encode_ledger(
            Ledger(1, (), receipt_dict["head_sha"], GENERATION),
            repo=receipt_dict["repository"],
            pr_number=receipt_dict["pr_number"],
        )
    return render_review_parts(
        collected=review_collected,
        lanes=[],
        issues=[],
        verdict=verdict,
        run_url=run_url or receipt_dict["run_url"],
        reviewed_sha=receipt_dict["head_sha"],
        hidden_marker=hidden_marker,
        receipt=receipt_dict,
    )[0]


def published_pair(**changes):
    receipt_dict = payload(**changes)
    review_collected = collected(
        scope=receipt_dict["scope"],
        kind="full-pr" if receipt_dict["scope"] in {"full-pr", "rebase"} else "commit-range",
        mode=receipt_dict["mode"],
    )
    body_text = render_body(receipt_dict, review_collected)
    canonical = canonical_receipt(parse_receipt(artifact(**changes)))
    return body_text, canonical, parse_receipt(canonical)


def test_receipt_marker_and_fixed_header_round_trip():
    body_text, canonical, receipt = published_pair()
    assert parse_review_receipt(body_text, canonical) == receipt
    assert receipt_digest(receipt) != receipt_digest(parse_receipt(artifact(profile="docs")))


@pytest.mark.parametrize("level", ["standard", "deep"])
def test_rebase_receipt_requires_full_diff_and_verify_mode(level):
    trigger = "manual" if level == "deep" else "baseline"
    body, canonical, receipt = published_pair(
        scope="rebase", mode="verify", level=level, trigger=trigger
    )
    assert parse_review_receipt(body, canonical) == receipt
    assert "**Scope:** `rebase` (full-pr)" in body
    with pytest.raises(SchemaError):
        parse_review_receipt(
            body.replace("`rebase` (full-pr)", "`rebase` (single-commit)"), canonical
        )
    with pytest.raises(SchemaError):
        parse_receipt(artifact(scope="rebase", mode="initial", level=level, trigger=trigger))


def test_rendered_initial_clean_verify_partial_and_deep_full_pr():
    initial_body, initial_canonical, initial = published_pair(mode="initial", scope="full-pr")
    assert parse_review_receipt(initial_body, initial_canonical) == initial
    assert initial.scope == "full-pr"

    verify_collected = collected(scope="latest-commit", kind="commit-range", mode="verify")
    partial_dict = payload(
        scope="latest-commit",
        mode="verify",
        verdict="partial",
        profile_satisfied=False,
        panel_status="error",
        level="standard",
        trigger="baseline",
        required_models=[],
        successful_models=[],
    )
    partial_body = render_body(partial_dict, verify_collected, verdict="partial", ledger=False)
    partial_canonical = canonical_receipt(parse_receipt(artifact(**partial_dict)))
    partial = parse_review_receipt(partial_body, partial_canonical)
    assert partial.scope == "latest-commit"
    assert partial.verdict == "partial"

    deep_body, deep_canonical, deep = published_pair(
        level="deep", trigger="manual", scope="full-pr"
    )
    assert parse_review_receipt(deep_body, deep_canonical) == deep
    assert deep.level == "deep"
    assert "**Scope:** `full-pr` (full-pr)" in deep_body


@pytest.mark.parametrize(
    "changes",
    [
        {"repository": "wrong repo"},
        {"pr_number": False},
        {"head_sha": "A" * 40},
        {"run_url": "https://elsewhere/x/actions/runs/2"},
        {"required_models": ["x/y", "x/y"]},
        {"successful_models": ["x/y"]},
        {"profile_satisfied": "true"},
        {"verdict": "bogus"},
        {"scope": "since-last-review"},
    ],
)
def test_receipt_rejects_bad_identity_types_or_invariants(changes):
    with pytest.raises(SchemaError):
        parse_receipt(artifact(**changes))


@pytest.mark.parametrize(
    "raw",
    [
        b'{"version":1,"version":1}',
        b'{"version":NaN}',
        b"{" + b"[" * 20 + b"0" + b"]" * 20 + b"}",
        b"{" + b"x" * (16 * 1024) + b"}",
    ],
)
def test_receipt_rejects_strict_json_failures(raw):
    with pytest.raises(SchemaError):
        parse_receipt(raw)


def test_body_rejects_tampered_ledger_receipt_and_header():
    body_text, canonical, receipt = published_pair()
    assert parse_review_receipt(body_text, canonical) == receipt

    lines = body_text.splitlines()
    bad_ledger = lines[:]
    bad_ledger[1] = bad_ledger[1][:-5] + "XXXXX -->"
    with pytest.raises(SchemaError, match="ledger"):
        parse_review_receipt("\n".join(bad_ledger), canonical)

    bad_marker = body_text.replace("v1:", "v1:x", 1)
    with pytest.raises(SchemaError):
        parse_review_receipt(bad_marker, canonical)

    bad_header = body_text.replace(f"**Commit:** `{HEAD}`", f"**Commit:** `{'f' * 40}`", 1)
    with pytest.raises(SchemaError):
        parse_review_receipt(bad_header, canonical)

    with pytest.raises(SchemaError, match="canonical JSON"):
        parse_review_receipt(body_text, canonical + b"\n")

    with pytest.raises(SchemaError, match="fixed header"):
        parse_review_receipt(body_text.replace(lines[1], ""), canonical)


@pytest.mark.parametrize("missing_head", [False, True])
def test_clean_receipt_requires_current_ledger_without_open_findings(missing_head):
    body_text, canonical, _ = published_pair()
    findings = (
        ()
        if missing_head
        else (LedgerFinding("r1-1", "bug", "a.py", 1, "Unresolved", "Evidence", "open"),)
    )
    marker = encode_ledger(
        Ledger(1, findings, "" if missing_head else HEAD, GENERATION),
        repo=REPO,
        pr_number=9,
    )
    lines = body_text.splitlines()
    lines[1] = marker
    with pytest.raises(SchemaError, match="ledger"):
        parse_review_receipt("\n".join(lines), canonical)


@pytest.mark.parametrize("has_open_finding", [False, True])
def test_clean_receipt_allows_settled_disputes_but_rejects_mixed_open_ledger(has_open_finding):
    body_text, canonical, receipt = published_pair(mode="verify")
    findings = [LedgerFinding("r1-1", "risk", "a.py", 1, "Rebutted", "Evidence", "disputed")]
    if has_open_finding:
        findings.append(LedgerFinding("r1-2", "bug", "b.py", 2, "Unresolved", "Evidence", "open"))
    lines = body_text.splitlines()
    lines[1] = encode_ledger(Ledger(3, tuple(findings), HEAD, GENERATION), repo=REPO, pr_number=9)
    body_text = "\n".join(lines)
    if has_open_finding:
        with pytest.raises(SchemaError, match="unresolved ledger findings"):
            parse_review_receipt(body_text, canonical)
    else:
        assert parse_review_receipt(body_text, canonical) == receipt


def test_gate_required_models_degraded_and_neutral_policy_base():
    receipt = parse_receipt(artifact(panel_status="degraded"))
    assert evaluate_receipt(receipt, **GATE_KWARGS).satisfied

    mismatch = dict(GATE_KWARGS)
    mismatch["repository"] = "other/repo"
    assert not evaluate_receipt(receipt, **mismatch).satisfied

    no_degraded = dict(GATE_KWARGS)
    no_degraded["allow_degraded"] = False
    assert not evaluate_receipt(receipt, **no_degraded).satisfied

    extra_models = dict(GATE_KWARGS)
    extra_models["required_models"] = ("vendor/required", "vendor/other")
    assert not evaluate_receipt(receipt, **extra_models).satisfied

    assert parse_receipt(artifact(policy_base_sha="f" * 40)).policy_digest == DIGEST
    standard = parse_receipt(artifact(level="standard", trigger="baseline"))
    assert not evaluate_receipt(standard, **GATE_KWARGS).satisfied


def test_evaluate_receipt_revalidates_before_gate():
    receipt = parse_receipt(artifact())
    tampered = replace(receipt, profile_satisfied=False)
    assert not evaluate_receipt(tampered, **GATE_KWARGS).satisfied
    assert evaluate_receipt(receipt, **GATE_KWARGS).satisfied
    assert not evaluate_receipt(tampered, **{**GATE_KWARGS, "minimum": 1}).satisfied


def request(run_id, kind, at, *, attempt=1, head=HEAD, origin=None):
    return AcceptedRequest(run_id, attempt, at, kind, head, DIGEST, REGISTRY, "code", origin)


def completion(run_id, at, *, attempt=1, head=HEAD, verdict="clean", level="deep"):
    deep = level == "deep"
    satisfied = verdict == "clean"
    receipt = parse_receipt(
        artifact(
            head_sha=head,
            verdict=verdict,
            profile_satisfied=satisfied,
            level=level,
            trigger="manual" if deep else "baseline",
            required_models=["vendor/required"] if deep else ["vendor/required"],
            successful_models=["vendor/required"] if satisfied else [],
            panel_status="complete" if satisfied else "error",
            run_url=f"https://github.example/{REPO}/actions/runs/{run_id}",
            run_attempt=attempt,
        )
    )
    return VerifiedRunReceipt(receipt, run_id, attempt, at)


def test_fold_carry_old_completion_and_new_completion():
    new_head = "f" * 40
    events = [request(1, "deep", 10), request(2, "carry", 20, head=new_head, origin=1)]
    assert fold_requests(events, [completion(1, 30)]) is not None
    assert fold_requests(events, [completion(1, 30), completion(2, 40, head=new_head)]) is None


def test_fold_standard_receipt_does_not_clear_deep_pending():
    events = [request(1, "deep", 10)]
    standard = completion(1, 20, level="standard")
    assert fold_requests(events, [standard]) is not None


@pytest.mark.parametrize("second_time", [19, 20, 21])
def test_new_deep_request_survives_older_completion_at_same_time(second_time):
    pending = fold_requests(
        [request(1, "deep", 10), request(2, "deep", second_time)],
        [completion(1, 20)],
    )
    assert pending is not None and pending.origin_run_id == 2
    assert (
        fold_requests(
            [request(1, "deep", 10), request(2, "deep", second_time)],
            [completion(1, 20), completion(2, 30)],
        )
        is None
    )


def test_fold_stale_head_completion_does_not_clear_carried():
    new_head = "f" * 40
    events = [request(1, "deep", 10), request(2, "carry", 20, head=new_head, origin=1)]
    assert fold_requests(events, [completion(2, 30, head=HEAD)]) is not None


def test_fold_cancel_requires_matching_origin():
    events = [request(1, "deep", 10)]
    with pytest.raises(SchemaError, match="cancel"):
        fold_requests(events + [request(2, "cancel", 20, origin=99)], [])


def test_fold_duplicate_deep_with_origin_preserves_origin():
    new_head = "f" * 40
    events = [
        request(1, "deep", 10),
        request(2, "deep", 20, head=new_head, origin=1),
    ]
    pending = fold_requests(events, [])
    assert pending is not None
    assert pending.origin_run_id == 1
    assert pending.head_sha == new_head
    assert pending.last_accepted_run_id == 2


def test_fold_rerun_attempt_does_not_clear_with_stale_completion():
    events = [request(1, "deep", 10, attempt=1), request(2, "carry", 20, attempt=2, origin=1)]
    assert fold_requests(events, [completion(1, 25, attempt=1)]) is not None
    assert (
        fold_requests(
            events,
            [completion(1, 25, attempt=1), completion(2, 30, attempt=2)],
        )
        is None
    )


def test_fold_cancel_failure_duplicates_and_bad_carry():
    events = [request(1, "deep", 10)]
    assert fold_requests(events, [completion(1, 20, verdict="error")]) is not None
    assert fold_requests(events + [request(2, "cancel", 20, origin=1)], []) is None
    assert fold_requests(events + events, []) is not None
    with pytest.raises(SchemaError):
        fold_requests([request(2, "carry", 10, origin=1)], [])


def test_fold_rejects_completion_run_binding_mismatch():
    with pytest.raises(SchemaError, match="run id"):
        fold_requests(
            [],
            [VerifiedRunReceipt(parse_receipt(artifact()), 99, 1, 10)],
        )
    with pytest.raises(SchemaError, match="run attempt"):
        fold_requests(
            [],
            [VerifiedRunReceipt(parse_receipt(artifact(run_attempt=1)), 42, 2, 10)],
        )


def test_fold_rejects_wrong_runtime_types():
    with pytest.raises(SchemaError):
        fold_requests([AcceptedRequest(1, 1, 10, "deep", 123, DIGEST, REGISTRY, "code", None)], [])
    bad_receipt = replace(parse_receipt(artifact()), head_sha=123)
    with pytest.raises(SchemaError):
        fold_requests([], [VerifiedRunReceipt(bad_receipt, 42, 1, 10)])


def test_fold_rejects_oversized_history():
    events = [request(index, "deep", index) for index in range(1, MAX_HISTORY_EVENTS + 2)]
    with pytest.raises(SchemaError, match="1000"):
        fold_requests(events, [])
