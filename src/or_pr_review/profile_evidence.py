"""Offline validation for review receipts and trusted request history.

This module deliberately does not authenticate GitHub data.  Its callers must
first establish that a run, job, and artifact came from the trusted workflow.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from or_pr_review.errors import SchemaError
from or_pr_review.loop import LEDGER_PREFIX, extract_ledger, has_ledger_marker

MAX_ARTIFACT_BYTES = 16 * 1024
MAX_BODY_BYTES = 60_000
MAX_HISTORY_EVENTS = 1000
_KEYS = (
    "version",
    "repository",
    "pr_number",
    "head_sha",
    "policy_base_sha",
    "policy_digest",
    "profile",
    "level",
    "trigger",
    "registry_digest",
    "context_sha256",
    "required_models",
    "successful_models",
    "panel_status",
    "profile_satisfied",
    "verdict",
    "scope",
    "mode",
    "run_url",
    "run_attempt",
)
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_GIT = re.compile(r"[0-9a-f]{40}\Z")
_PROFILE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_REPOSITORY = re.compile(r"[^/\s]{1,100}/[^/\s]{1,100}\Z")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*(?::[A-Za-z0-9._-]+)?\Z")
_RUN_ID = re.compile(r"/actions/runs/([1-9][0-9]*)\Z")


@dataclass(frozen=True)
class ReviewReceipt:
    version: int
    repository: str
    pr_number: int
    head_sha: str
    policy_base_sha: str
    policy_digest: str
    profile: str
    level: str
    trigger: str
    registry_digest: str
    context_sha256: str
    required_models: tuple[str, ...]
    successful_models: tuple[str, ...]
    panel_status: str
    profile_satisfied: bool
    verdict: str
    scope: str
    mode: str
    run_url: str
    run_attempt: int


@dataclass(frozen=True)
class GateDecision:
    satisfied: bool
    reason: str


@dataclass(frozen=True)
class AcceptedRequest:
    """An event admitted only after external trusted-workflow provenance checks."""

    run_id: int
    run_attempt: int
    accepted_at_ms: int
    kind: str
    head_sha: str
    policy_digest: str
    registry_digest: str
    profile: str
    origin_run_id: int | None


@dataclass(frozen=True)
class VerifiedRunReceipt:
    """A receipt paired with a run only after external artifact provenance checks."""

    receipt: ReviewReceipt
    run_id: int
    run_attempt: int
    completed_at_ms: int


@dataclass(frozen=True)
class PendingRequest:
    origin_run_id: int
    head_sha: str
    policy_digest: str
    registry_digest: str
    profile: str
    last_accepted_run_id: int
    last_accepted_run_attempt: int


def _fail(message: str) -> SchemaError:
    return SchemaError(f"review receipt: {message}")


def _text(value: Any, what: str, limit: int = 512) -> str:
    if type(value) is not str:
        raise _fail(f"{what} must be a string")
    try:
        if len(value.encode("utf-8", "strict")) > limit:
            raise _fail(f"{what} is too long")
    except UnicodeError as exc:
        raise _fail(f"{what} is not valid Unicode") from exc
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise _fail(f"{what} is not valid Unicode")
    return value


def _slug(value: Any, what: str) -> str:
    text = _text(value, what, 200)
    if text != text.strip() or not _MODEL.fullmatch(text):
        raise _fail(f"{what} is not a model slug")
    return text


def _models(value: Any, what: str) -> tuple[str, ...]:
    if type(value) is not list or len(value) > 4:
        raise _fail(f"{what} must be a list of at most four models")
    models = tuple(_slug(item, what) for item in value)
    if len(models) != len(set(models)):
        raise _fail(f"{what} must not contain duplicates")
    return models


def _strict_json(raw: bytes | str) -> dict[str, Any]:
    if isinstance(raw, bytes):
        if len(raw) > MAX_ARTIFACT_BYTES:
            raise _fail("artifact exceeds 16 KiB")
        try:
            text = raw.decode("utf-8", "strict")
        except UnicodeError as exc:
            raise _fail("artifact is not UTF-8") from exc
    elif isinstance(raw, str):
        text = _text(raw, "artifact", MAX_ARTIFACT_BYTES)
    else:
        raise _fail("artifact must be bytes or text")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _fail(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=no_duplicates,
            parse_constant=lambda name: (_ for _ in ()).throw(ValueError(name)),
        )
    except SchemaError:
        raise
    except (json.JSONDecodeError, RecursionError, UnicodeError, ValueError) as exc:
        raise _fail("artifact is not strict JSON") from exc

    def depth(item: Any, current: int = 0) -> None:
        if current > 16:
            raise _fail("artifact is nested too deeply")
        if type(item) is dict:
            for key, child in item.items():
                _text(key, "JSON key")
                depth(child, current + 1)
        elif type(item) is list:
            for child in item:
                depth(child, current + 1)
        elif type(item) is str:
            _text(item, "JSON text", MAX_ARTIFACT_BYTES)

    depth(value)
    if type(value) is not dict:
        raise _fail("artifact must be an object")
    return value


def _url(value: Any, repository: str) -> str:
    text = _text(value, "run_url")
    if not re.fullmatch(
        rf"https://[A-Za-z0-9][A-Za-z0-9.-]*/{re.escape(repository)}/actions/runs/[1-9][0-9]*",
        text,
    ):
        raise _fail("run_url is not a source workflow-run URL for the repository")
    return text


def _run_id_from_url(url: str) -> int:
    match = _RUN_ID.search(url)
    if match is None:
        raise _fail("run_url is not a source workflow-run URL for the repository")
    return int(match.group(1))


def _allowed_scope_lines(scope: str) -> set[str]:
    base = f"**Scope:** `{scope}`"
    if scope == "full-pr":
        return {f"{base} (full-pr)"}
    return {f"{base} (commit-range)", f"{base} (single-commit)"}


def parse_receipt(raw: bytes | str) -> ReviewReceipt:
    value = _strict_json(raw)
    if set(value) != set(_KEYS):
        raise _fail("artifact keys are not exactly receipt v1 keys")
    if type(value["version"]) is not int or value["version"] != 1:
        raise _fail("unsupported receipt version")
    repository = _text(value["repository"], "repository", 201)
    if not _REPOSITORY.fullmatch(repository):
        raise _fail("repository must be owner/name")
    if type(value["pr_number"]) is not int or value["pr_number"] < 1:
        raise _fail("pr_number must be positive")
    for key in ("head_sha",):
        if type(value[key]) is not str or not _GIT.fullmatch(value[key]):
            raise _fail(f"{key} must be a lowercase full SHA")
    policy_digest = _text(value["policy_digest"], "policy_digest", 64)
    policy_base_sha = _text(value["policy_base_sha"], "policy_base_sha", 40)
    if policy_digest:
        if not _SHA.fullmatch(policy_digest) or not _GIT.fullmatch(policy_base_sha):
            raise _fail("configured policy requires lowercase digest and base SHA")
    elif policy_base_sha and not _GIT.fullmatch(policy_base_sha):
        raise _fail("policy_base_sha must be empty or a lowercase full SHA")
    for key in ("registry_digest", "context_sha256"):
        if type(value[key]) is not str or not _SHA.fullmatch(value[key]):
            raise _fail(f"{key} must be a lowercase SHA-256")
    profile = _text(value["profile"], "profile", 64)
    if not _PROFILE.fullmatch(profile):
        raise _fail("profile is invalid")
    level, trigger = value["level"], value["trigger"]
    if (
        type(level) is not str
        or type(trigger) is not str
        or (level, trigger)
        not in {
            ("standard", "baseline"),
            ("deep", "manual"),
            ("deep", "policy"),
        }
    ):
        raise _fail("level and trigger are incoherent")
    required = _models(value["required_models"], "required_models")
    successful = _models(value["successful_models"], "successful_models")
    panel, verdict = value["panel_status"], value["verdict"]
    if type(panel) is not str or panel not in {"complete", "degraded", "required_missing", "error"}:
        raise _fail("panel_status is invalid")
    if type(value["profile_satisfied"]) is not bool:
        raise _fail("profile_satisfied must be a boolean")
    satisfied = value["profile_satisfied"]
    if type(verdict) is not str or verdict not in {"clean", "issues", "partial", "error"}:
        raise _fail("verdict is invalid")
    if (not satisfied and verdict == "clean") or (
        satisfied and (verdict in {"partial", "error"} or panel in {"required_missing", "error"})
    ):
        raise _fail("satisfaction conflicts with verdict or panel status")
    if satisfied and (not successful or not set(required).issubset(successful)):
        raise _fail("satisfied receipt lacks required successful models")
    if satisfied and level == "deep" and not required:
        raise _fail("deep receipt cannot be satisfied without required models")
    scope, mode = value["scope"], value["mode"]
    if type(scope) is not str or scope not in {"full-pr", "latest-commit"}:
        raise _fail("scope is invalid")
    if type(mode) is not str or mode not in {"initial", "verify"}:
        raise _fail("mode is invalid")
    if level == "deep" and scope != "full-pr" or mode == "initial" and scope != "full-pr":
        raise _fail("level or mode requires full-pr scope")
    run_url = _url(value["run_url"], repository)
    if type(value["run_attempt"]) is not int or not 1 <= value["run_attempt"] <= 1000:
        raise _fail("run_attempt must be from 1 through 1000")
    result = {key: value[key] for key in _KEYS}
    result.update(
        repository=repository,
        policy_digest=policy_digest,
        policy_base_sha=policy_base_sha,
        profile=profile,
        required_models=required,
        successful_models=successful,
        run_url=run_url,
    )
    return ReviewReceipt(**result)


def canonical_receipt(receipt: ReviewReceipt) -> bytes:
    if type(receipt) is not ReviewReceipt:
        raise _fail("canonical receipt requires ReviewReceipt")
    # Reparse so hand-constructed dataclasses receive exactly the same checks.
    validated = parse_receipt(json.dumps(asdict(receipt), separators=(",", ":"), allow_nan=False))
    return json.dumps(
        asdict(validated), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def receipt_digest(receipt: ReviewReceipt) -> str:
    return hashlib.sha256(canonical_receipt(receipt)).hexdigest()


def _header_lines(body: str, receipt: ReviewReceipt) -> int:
    lines = body.splitlines()
    title = "## OpenRouter pull-request review"
    scope_options = _allowed_scope_lines(receipt.scope)
    verdict_line = f"**Verdict:** `{receipt.verdict}`"
    mode_line = f"**Mode:** `{receipt.mode}`"
    commit_line = f"**Commit:** `{receipt.head_sha}`"

    def with_ledger() -> int | None:
        if (
            len(lines) < 7
            or lines[0] != title
            or not lines[1].startswith(LEDGER_PREFIX)
            or not lines[1].endswith(" -->")
            or lines[2] != ""
            or lines[3] != verdict_line
            or lines[4] not in scope_options
            or lines[5] != mode_line
            or lines[6] != commit_line
        ):
            return None
        ledger = extract_ledger(body, repo=receipt.repository, pr_number=receipt.pr_number)
        if ledger is None:
            raise _fail("review ledger marker does not decode for this repository")
        if ledger.reviewed_sha != receipt.head_sha:
            raise _fail("review ledger head SHA does not match receipt")
        if receipt.verdict == "clean" and ledger.findings:
            raise _fail("clean receipt conflicts with unresolved ledger findings")
        return 6

    def without_ledger() -> int | None:
        if (
            len(lines) < 6
            or lines[0] != title
            or lines[1] != ""
            or lines[2] != verdict_line
            or lines[3] not in scope_options
            or lines[4] != mode_line
            or lines[5] != commit_line
        ):
            return None
        if has_ledger_marker(body):
            raise _fail("partial review body must not publish loop ledger state")
        return 5

    if receipt.verdict in {"clean", "issues"}:
        index = with_ledger()
        if index is None:
            raise _fail("review body fixed header does not match receipt")
        return index
    if receipt.verdict in {"partial", "error"}:
        index = without_ledger()
        if index is None:
            raise _fail("review body fixed header does not match receipt")
        return index
    raise _fail("review body fixed header does not match receipt")


def parse_review_receipt(body: str, artifact: bytes | str) -> ReviewReceipt:
    body = _text(body, "body", MAX_BODY_BYTES)
    receipt = parse_receipt(artifact)
    commit_index = _header_lines(body, receipt)
    marker_prefix = "<!-- openrouter-review-plan:v1:"
    marker_lines = [
        (index, line)
        for index, line in enumerate(body.splitlines())
        if line.startswith(marker_prefix)
    ]
    if len(marker_lines) != 1:
        raise _fail("review body must contain exactly one receipt marker")
    index, marker = marker_lines[0]
    lanes = next((i for i, line in enumerate(body.splitlines()) if line == "### Lanes"), None)
    if lanes is None or not commit_index < index < lanes or not marker.endswith(" -->"):
        raise _fail("receipt marker is not after Commit and before Lanes")
    encoded = marker[len(marker_prefix) : -4]
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _fail("receipt marker is not strict base64") from exc
    canonical = canonical_receipt(receipt)
    if decoded != canonical:
        raise _fail("receipt marker does not equal canonical artifact")
    artifact_bytes = artifact if isinstance(artifact, bytes) else artifact.encode("utf-8", "strict")
    if artifact_bytes != canonical:
        raise _fail("receipt artifact is not canonical JSON")
    if body.count(f"[Workflow run]({receipt.run_url})") != 1:
        raise _fail("review body must contain the exact workflow run link")
    return receipt


def evaluate_receipt(
    receipt: ReviewReceipt,
    *,
    repository: str,
    pr_number: int,
    head_sha: str,
    policy_digest: str,
    registry_digest: str,
    profile: str,
    minimum: str,
    required_models: tuple[str, ...],
    allow_degraded: bool = True,
) -> GateDecision:
    if type(receipt) is not ReviewReceipt:
        return GateDecision(False, "receipt is not validated")
    try:
        receipt = parse_receipt(canonical_receipt(receipt))
    except SchemaError:
        return GateDecision(False, "receipt is not validated")
    if type(repository) is not str or not _REPOSITORY.fullmatch(repository):
        return GateDecision(False, "repository mismatch")
    if type(pr_number) is not int or pr_number < 1:
        return GateDecision(False, "pull request mismatch")
    if type(head_sha) is not str or not _GIT.fullmatch(head_sha):
        return GateDecision(False, "head SHA mismatch")
    if type(policy_digest) is not str:
        return GateDecision(False, "policy digest mismatch")
    if type(registry_digest) is not str or not _SHA.fullmatch(registry_digest):
        return GateDecision(False, "registry digest mismatch")
    if type(profile) is not str or not _PROFILE.fullmatch(profile):
        return GateDecision(False, "profile mismatch")
    checks = (
        (receipt.repository == repository, "repository mismatch"),
        (receipt.pr_number == pr_number, "pull request mismatch"),
        (receipt.head_sha == head_sha, "head SHA mismatch"),
        (receipt.policy_digest == policy_digest, "policy digest mismatch"),
        (receipt.registry_digest == registry_digest, "registry digest mismatch"),
        (receipt.profile == profile, "profile mismatch"),
    )
    for valid, reason in checks:
        if not valid:
            return GateDecision(False, reason)
    if type(minimum) is not str or minimum not in {"standard", "deep"}:
        return GateDecision(False, "invalid minimum level")
    if receipt.level == "standard" and minimum == "deep":
        return GateDecision(False, "standard receipt cannot satisfy deep minimum")
    if receipt.verdict != "clean" or not receipt.profile_satisfied:
        return GateDecision(False, "receipt is not a satisfied clean review")
    if type(allow_degraded) is not bool:
        return GateDecision(False, "allow_degraded must be a boolean")
    if receipt.panel_status == "degraded" and not allow_degraded:
        return GateDecision(False, "degraded panel is not allowed")
    if receipt.panel_status not in ({"complete", "degraded"} if allow_degraded else {"complete"}):
        return GateDecision(False, "panel status is not allowed")
    if type(required_models) is not tuple or any(type(item) is not str for item in required_models):
        return GateDecision(False, "expected required models are invalid")
    if not set(required_models).issubset(receipt.required_models):
        return GateDecision(False, "receipt removed an expected required model")
    if not set(required_models).issubset(receipt.successful_models):
        return GateDecision(False, "an expected required model was not successful")
    return GateDecision(True, "receipt satisfies current review requirements")


def _valid_request(event: AcceptedRequest) -> None:
    if type(event) is not AcceptedRequest:
        raise _fail("invalid accepted request")
    if type(event.run_id) is not int or event.run_id < 1:
        raise _fail("invalid accepted request")
    if type(event.run_attempt) is not int or not 1 <= event.run_attempt <= 1000:
        raise _fail("invalid accepted request")
    if type(event.accepted_at_ms) is not int or event.accepted_at_ms < 0:
        raise _fail("invalid accepted request")
    if type(event.kind) is not str or event.kind not in {"deep", "cancel", "carry"}:
        raise _fail("invalid accepted request")
    if type(event.head_sha) is not str or not _GIT.fullmatch(event.head_sha):
        raise _fail("invalid accepted request")
    if type(event.policy_digest) is not str or not (
        event.policy_digest == "" or _SHA.fullmatch(event.policy_digest)
    ):
        raise _fail("invalid accepted request")
    if type(event.registry_digest) is not str or not _SHA.fullmatch(event.registry_digest):
        raise _fail("invalid accepted request")
    if type(event.profile) is not str or not _PROFILE.fullmatch(event.profile):
        raise _fail("invalid accepted request")
    if event.origin_run_id is not None and (
        type(event.origin_run_id) is not int or event.origin_run_id < 1
    ):
        raise _fail("invalid accepted request")


def _valid_completion(item: VerifiedRunReceipt) -> None:
    if (
        type(item) is not VerifiedRunReceipt
        or type(item.run_id) is not int
        or item.run_id < 1
        or type(item.run_attempt) is not int
        or not 1 <= item.run_attempt <= 1000
        or type(item.completed_at_ms) is not int
        or item.completed_at_ms < 0
        or type(item.receipt) is not ReviewReceipt
    ):
        raise _fail("invalid verified completion")
    receipt = parse_receipt(canonical_receipt(item.receipt))
    if receipt.run_attempt != item.run_attempt:
        raise _fail("completion run attempt does not match receipt")
    if _run_id_from_url(receipt.run_url) != item.run_id:
        raise _fail("completion run id does not match receipt")


def fold_requests(
    events: Iterable[AcceptedRequest], completions: Iterable[VerifiedRunReceipt]
) -> PendingRequest | None:
    requests: list[AcceptedRequest] = []
    for event in events:
        if len(requests) >= MAX_HISTORY_EVENTS:
            raise _fail("accepted request history exceeds 1000 events")
        requests.append(event)
    finished: list[VerifiedRunReceipt] = []
    for item in completions:
        if len(finished) >= MAX_HISTORY_EVENTS:
            raise _fail("completion history exceeds 1000 events")
        finished.append(item)

    request_by_key: dict[tuple[int, int], AcceptedRequest] = {}
    for event in requests:
        _valid_request(event)
        key = (event.run_id, event.run_attempt)
        previous = request_by_key.get(key)
        if previous is not None:
            if previous != event:
                raise _fail("contradictory duplicate accepted request")
            continue
        request_by_key[key] = event

    completion_by_key: dict[tuple[int, int], VerifiedRunReceipt] = {}
    for item in finished:
        _valid_completion(item)
        key = (item.run_id, item.run_attempt)
        previous = completion_by_key.get(key)
        if previous is not None:
            if previous != item:
                raise _fail("contradictory duplicate completion")
            continue
        completion_by_key[key] = item

    for key, event in request_by_key.items():
        completion = completion_by_key.get(key)
        if completion is not None and completion.completed_at_ms < event.accepted_at_ms:
            raise _fail("completion precedes its accepted request")

    timeline: list[tuple[int, int, int, int, object]] = []
    timeline.extend(
        (event.accepted_at_ms, 0, event.run_id, event.run_attempt, event)
        for event in request_by_key.values()
    )
    timeline.extend(
        (item.completed_at_ms, 1, item.run_id, item.run_attempt, item)
        for item in completion_by_key.values()
    )
    pending: PendingRequest | None = None
    for _time, phase, _run_id, _attempt, item in sorted(timeline):
        if phase == 0:
            event = item
            assert isinstance(event, AcceptedRequest)
            if event.kind == "deep":
                if event.origin_run_id is not None:
                    if pending is None or event.origin_run_id != pending.origin_run_id:
                        raise _fail("deep request has an invalid order")
                    pending = PendingRequest(
                        pending.origin_run_id,
                        event.head_sha,
                        event.policy_digest,
                        event.registry_digest,
                        event.profile,
                        event.run_id,
                        event.run_attempt,
                    )
                else:
                    # A fresh authorized request supersedes the outstanding
                    # one, even when GitHub gives both events the same time.
                    # An older completion cannot clear this new obligation.
                    pending = PendingRequest(
                        event.run_id,
                        event.head_sha,
                        event.policy_digest,
                        event.registry_digest,
                        event.profile,
                        event.run_id,
                        event.run_attempt,
                    )
            elif event.kind == "carry":
                if pending is None or event.origin_run_id != pending.origin_run_id:
                    raise _fail("carry does not reference the currently pending request")
                pending = PendingRequest(
                    pending.origin_run_id,
                    event.head_sha,
                    event.policy_digest,
                    event.registry_digest,
                    event.profile,
                    event.run_id,
                    event.run_attempt,
                )
            else:
                if pending is None or event.origin_run_id != pending.origin_run_id:
                    raise _fail("cancel does not reference the currently pending request")
                pending = None
        else:
            completion = item
            assert isinstance(completion, VerifiedRunReceipt)
            if pending is None or (
                completion.run_id,
                completion.run_attempt,
            ) != (pending.last_accepted_run_id, pending.last_accepted_run_attempt):
                continue
            receipt = completion.receipt
            matches_pending = (
                receipt.head_sha,
                receipt.policy_digest,
                receipt.registry_digest,
                receipt.profile,
            ) == (
                pending.head_sha,
                pending.policy_digest,
                pending.registry_digest,
                pending.profile,
            )
            if (
                receipt.level == "deep"
                and receipt.verdict == "clean"
                and receipt.profile_satisfied
                and receipt.panel_status in {"complete", "degraded"}
                and matches_pending
            ):
                pending = None
    return pending
