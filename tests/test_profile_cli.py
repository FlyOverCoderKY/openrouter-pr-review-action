"""Offline contract checks for prepared profile entry validation and artifacts."""

from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from or_pr_review import cli
from or_pr_review.collect import CollectedReview
from or_pr_review.errors import ActionError, SchemaError
from or_pr_review.harness import DEFAULT_LANE_TIMEOUT_SECONDS
from or_pr_review.loop import LedgerFinding, LoopState, extract_ledger
from or_pr_review.profile_evidence import parse_review_receipt
from or_pr_review.publish import render_review_parts
from or_pr_review.review_context import (
    PreparedExecution,
    freeze_context,
    freeze_runtime,
    restore_context,
)
from or_pr_review.review_plan import ReviewLane, ReviewPlan
from or_pr_review.schema import (
    SCHEMA_VERSION,
    Finding,
    LaneResult,
    failed_lane,
    parse_lane_artifact,
)
from test_policy_integration import HEAD, sample

REPO = "owner/repo"
RUN_URL = f"https://github.com/{REPO}/actions/runs/42"
HEAD_SHA = HEAD
BASE_SHA = "b" * 40

REGISTRY = """{
  "version": 1,
  "profiles": {
    "code": {
      "standard": {
        "lanes": [{"model": "vendor/required", "required": true}],
        "lane_timeout_seconds": 600,
        "job_budget_seconds": 1320
      },
      "deep": {
        "lanes": [
          {"model": "vendor/required", "required": true},
          {"model": "vendor/deep", "required": false}
        ],
        "lane_timeout_seconds": 900,
        "job_budget_seconds": 1320
      }
    }
  }
}"""

DEEP_TWO_LANE = """{
  "version": 1,
  "profiles": {
    "code": {
      "standard": {
        "lanes": [{"model": "vendor/required", "required": true}],
        "lane_timeout_seconds": 600,
        "job_budget_seconds": 1320
      },
      "deep": {
        "lanes": [
          {"model": "vendor/required", "required": true},
          {"model": "vendor/optional", "required": false}
        ],
        "judge_model": "vendor/judge",
        "lane_timeout_seconds": 900,
        "job_budget_seconds": 1320
      }
    }
  }
}"""


def _profile_env(tmp_path, **extra: str) -> dict[str, str]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    env = {
        "REVIEW_PROFILES": REGISTRY,
        "REVIEW_LEVEL": "deep",
        "GITHUB_REPOSITORY": REPO,
        "PR_NUMBER": "9",
        "HEAD_SHA": HEAD_SHA,
        "RUN_URL": RUN_URL,
        "GITHUB_RUN_ATTEMPT": "1",
        "STATUS_COMMENTS": "false",
        "GITHUB_OUTPUT": str(tmp_path / "outputs.txt"),
        "RUNNER_TEMP": str(tmp_path),
        "WORK": str(tmp_path / "work"),
        "SOURCE_WORKSPACE": str(tmp_path / "workspace"),
        "OPENROUTER_API_KEY": "sk-test",
        "ALL_LANE_RESULTS_DIR": str(tmp_path / "lanes"),
        "LANE_RESULTS_DIR": str(tmp_path / "lanes"),
    }
    env.update(extra)
    return env


def _collected(*, mode: str = "initial", loop_mode: str = "initial") -> CollectedReview:
    return replace(sample(), pr_number=9, mode=loop_mode)


def _loop(*, mode: str = "initial", generation: str = "") -> LoopState:
    if mode == "verify":
        return LoopState(
            "verify",
            2,
            (
                LedgerFinding(
                    "r1-1",
                    "bug",
                    "storage/a.py",
                    1,
                    "Prior bug",
                    "Evidence",
                    "open",
                    ("vendor/required",),
                ),
            ),
            generation or "1234567890ab",
        )
    return LoopState("initial", 1)


class FakeGitHub:
    def __init__(self, *, head: str = HEAD_SHA, replies: list[tuple[str, str]] | None = None):
        self.head = head
        self.replies = replies or []
        self.posted: list[dict[str, Any]] = []

    def pr_view(self, _number: int) -> dict[str, str]:
        return {"headRefOid": self.head}

    def create_review(self, _number: int, body: str, commit: str, **kwargs: Any) -> dict[str, str]:
        self.posted.append({"body": body, "commit": commit, **kwargs})
        return {"html_url": "https://github.com/review/1"}

    def create_issue_comment(self, *_args: Any) -> None:
        return None

    def list_bot_review_bodies(self, *_args: Any) -> list[str]:
        return []

    def list_finding_replies(self, *_args: Any, **kwargs: Any) -> list[tuple[str, str]]:
        return self.replies

    def list_recent_issue_comments(self, *_args: Any) -> list[str]:
        return []


def _lane_ok(model: str, *, finding: Finding | None = None) -> LaneResult:
    findings = [finding] if finding is not None else []
    return LaneResult(SCHEMA_VERSION, True, model, findings)


def _install_judge_skip(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "_resolve_issues",
        lambda *_a, **_k: cli.JudgeOutcome([], "skipped", None, False, []),
    )


def _install_collect(monkeypatch, *, loop: LoopState | None = None, replies: str = "") -> None:
    state = loop or _loop()
    collected = _collected(loop_mode=state.mode)

    def collect(_env: dict[str, str], with_replies: bool = True):
        if with_replies and state.mode == "verify":
            return collected, state, replies
        return collected, state, replies if with_replies else ""

    monkeypatch.setattr(cli, "_collect_with_loop", collect)
    monkeypatch.setattr(cli, "_collect", lambda _env: collected)
    monkeypatch.setattr(cli, "_prepare_workspace", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_maybe_status", lambda *_a, **_k: None)


def _run_setup(
    tmp_path, monkeypatch, env: dict[str, str] | None = None
) -> tuple[dict[str, Any], str]:
    settings = env or _profile_env(tmp_path)
    assert cli.main(["setup"], settings) == 0
    context_path = Path(
        settings.get("REVIEW_CONTEXT_OUTPUT_FILE") or (tmp_path / "lanes" / "review-context.json")
    )
    if not context_path.is_file():
        outputs = (tmp_path / "outputs.txt").read_text(encoding="utf-8")
        for line in outputs.splitlines():
            if line.startswith("review_context_file="):
                context_path = Path(line.split("=", 1)[1])
    envelope = json.loads(context_path.read_text(encoding="utf-8"))
    return envelope, envelope["sha256"]


def _context_env(tmp_path, envelope: dict[str, Any], digest: str, **extra: str) -> dict[str, str]:
    context_path = tmp_path / "lanes" / "review-context.json"
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text(json.dumps(envelope, sort_keys=True) + "\n", encoding="utf-8")
    return _profile_env(
        tmp_path,
        REVIEW_CONTEXT_FILE=str(context_path),
        REVIEW_CONTEXT_SHA256=digest,
        **extra,
    )


def test_profile_configuration_rejects_legacy_model_overrides_before_collection() -> None:
    with pytest.raises(ActionError, match="cannot be mixed"):
        cli._validate_profile_inputs({"REVIEW_PROFILES": REGISTRY, "MODELS": "vendor/other"})


def test_deep_without_registry_rejects_before_collection() -> None:
    with pytest.raises(ActionError, match="requires a configured deep"):
        cli._validate_profile_inputs({"REVIEW_LEVEL": "deep"})


def test_prepared_lane_artifact_round_trips_provenance() -> None:
    lane = LaneResult(SCHEMA_VERSION, True, "vendor/required", [])
    lane.lane_index = 0
    lane.required = True
    lane.context_sha256 = "a" * 64
    restored = parse_lane_artifact(lane.to_dict())
    assert (restored.lane_index, restored.required, restored.context_sha256) == (
        0,
        True,
        "a" * 64,
    )


def test_prepared_lane_artifact_rejects_unrecognized_fields() -> None:
    with pytest.raises(SchemaError, match="unexpected keys"):
        parse_lane_artifact(
            {
                "schema_version": SCHEMA_VERSION,
                "ok": True,
                "model": "vendor/required",
                "findings": [],
                "error": None,
                "untrusted": "ignored-by-old-parser",
            }
        )


def test_prepared_all_collector_starts_every_lane_before_any_finishes(tmp_path) -> None:
    """The compatibility all-role panel keeps the parallel matrix behavior."""
    plan = ReviewPlan(
        "code",
        "deep",
        "manual",
        (ReviewLane("vendor/one", True), ReviewLane("vendor/two", False)),
        "vendor/judge",
        "",
        0,
        60,
        300,
        "a" * 64,
        False,
    )
    context = cli.PreparedContext(
        SimpleNamespace(
            collected=SimpleNamespace(head_sha="b" * 40),
            execution=SimpleNamespace(plan=plan),
        ),
        "c" * 64,
        {"version": 3},
    )
    barrier = threading.Barrier(2, timeout=1)
    events: list[tuple[str, int]] = []

    def run_one(index: int) -> LaneResult:
        events.append(("started", index))
        barrier.wait()
        events.append(("finished", index))
        return LaneResult(SCHEMA_VERSION, True, plan.lanes[index].model, [])

    lanes = cli._collect_all_prepared_lanes(plan, context, tmp_path, run_one, 1)

    assert [lane.model for lane in lanes] == ["vendor/one", "vendor/two"]
    assert {index for event, index in events if event == "started"} == {0, 1}
    assert all(
        event != "finished" or len({i for kind, i in events[:position] if kind == "started"}) == 2
        for position, (event, _index) in enumerate(events)
    )


def test_prepared_json_reader_rejects_duplicate_keys_and_oversized_files(tmp_path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(SchemaError, match="duplicate"):
        cli._read_bounded_json(duplicate, what="test artifact")

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * (16 * 1024 * 1024 + 1))
    with pytest.raises(SchemaError, match="16 MiB"):
        cli._read_bounded_json(oversized, what="test artifact")


@pytest.mark.parametrize(
    ("lanes", "frozen", "expected"),
    [
        ((ReviewLane("vendor/one", True),), "true", True),
        ((ReviewLane("vendor/one", True), ReviewLane("vendor/two", False)), "false", False),
    ],
)
def test_legacy_frozen_judge_override_wins_over_lane_count(lanes, frozen, expected) -> None:
    plan = SimpleNamespace(lanes=lanes)
    context = SimpleNamespace(
        execution=SimpleNamespace(plan=plan, runtime_json=json.dumps({"JUDGE_NEEDED": frozen}))
    )
    assert cli._prepared_judge_needed(context) is expected


def test_receipt_file_is_the_exact_hidden_marker_payload(tmp_path, monkeypatch) -> None:
    receipt = {"profile": "code", "profile_satisfied": False, "version": 1}
    monkeypatch.setattr(cli, "_ACTIVE_ENV", {})
    path = cli._write_review_receipt({"ALL_LANE_RESULTS_DIR": str(tmp_path)}, receipt)
    canonical = path.read_text(encoding="utf-8").strip()
    assert canonical == json.dumps(receipt, sort_keys=True, separators=(",", ":"))

    body = render_review_parts(
        collected=sample(), lanes=[], issues=[], verdict="partial", receipt=receipt
    )[0]
    marker = next(
        line for line in body.splitlines() if line.startswith("<!-- openrouter-review-plan:")
    )
    encoded = marker.split(":", 2)[2].removesuffix(" -->")
    assert base64.b64decode(encoded).decode("utf-8") == canonical


def test_paired_context_inputs_reject_before_collection(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "_best_effort_incomplete", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_collect_with_loop", lambda *a, **k: pytest.fail("must not collect"))
    env = _profile_env(tmp_path, REVIEW_CONTEXT_FILE=str(tmp_path / "ctx.json"))
    assert cli.main(["all"], env) == 1
    env = _profile_env(tmp_path, REVIEW_CONTEXT_SHA256="a" * 64)
    assert cli.main(["lane"], env) == 1


def test_setup_writes_context_to_output_destination(tmp_path, monkeypatch) -> None:
    _install_collect(monkeypatch)
    destination = tmp_path / "nested" / "frozen-context.json"
    env = _profile_env(tmp_path, REVIEW_CONTEXT_OUTPUT_FILE=str(destination))
    assert cli.main(["setup"], env) == 0
    assert destination.is_file()
    assert json.loads(destination.read_text(encoding="utf-8"))["version"] == 3


@pytest.mark.parametrize("post_fails", [False, True])
def test_finish_receipt_producer_round_trips_through_profile_evidence(
    tmp_path, monkeypatch, post_fails
) -> None:
    collected = _collected()
    loop = _loop()
    plan = ReviewPlan(
        "code",
        "deep",
        "manual",
        (ReviewLane("vendor/required", True),),
        "vendor/judge",
        "",
        50,
        600,
        1320,
        "d" * 64,
        False,
    )
    execution = PreparedExecution(
        plan,
        freeze_runtime({}),
        "",
        int(time.time() * 1000),
        int(time.time() * 1000) + 1_320_000,
        RUN_URL,
        2,
    )
    envelope = freeze_context(REPO, collected, loop, 50, execution=execution)
    context = cli.PreparedContext(restore_context(envelope), envelope["sha256"], envelope)
    github = FakeGitHub()
    monkeypatch.setattr(cli, "_github", lambda _env: github)
    monkeypatch.setattr(
        cli,
        "_resolve_issues",
        lambda *_a, **_k: cli.JudgeOutcome([], "skipped", None, False, []),
    )
    lane = _lane_ok("vendor/required")
    lane.head_sha = collected.head_sha
    lane.review_context = envelope
    env = _profile_env(tmp_path, GITHUB_RUN_ATTEMPT="2")
    frozen_env = cli._frozen_execution_env(env, context.context)
    monkeypatch.setattr(cli, "_ACTIVE_ENV", frozen_env)
    if post_fails:

        def failed_post(*args, **kwargs):
            raise ActionError("GitHub is unavailable")

        monkeypatch.setattr(github, "create_review", failed_post)
        with pytest.raises(ActionError, match="failed to post"):
            cli._finish(
                frozen_env, [lane], collected=collected, loop=loop, prepared_context=context
            )
        assert not (tmp_path / "lanes" / "review-receipt.json").exists()
        return
    assert (
        cli._finish(
            frozen_env,
            [lane],
            collected=collected,
            loop=loop,
            prepared_context=context,
        )
        == 0
    )
    artifact = (tmp_path / "lanes" / "review-receipt.json").read_text(encoding="utf-8")
    receipt = parse_review_receipt(github.posted[0]["body"], artifact)
    assert receipt.repository == REPO
    assert receipt.pr_number == collected.pr_number
    assert receipt.verdict == "clean"
    assert receipt.scope == collected.plan.scope
    assert receipt.mode == loop.mode
    assert receipt.run_attempt == 2
    assert artifact == json.dumps(json.loads(artifact), sort_keys=True, separators=(",", ":"))


def test_prepared_setup_lane_judge_collects_once(tmp_path, monkeypatch) -> None:
    calls = {"collect": 0, "lane": 0}

    def collect(_env, with_replies=True):
        calls["collect"] += 1
        return _collected(), _loop(), ""

    monkeypatch.setattr(cli, "_collect_with_loop", collect)
    monkeypatch.setattr(cli, "_prepare_workspace", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_maybe_status", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_github", lambda _env: FakeGitHub())

    def invoke(_env, model, *_a, **_k):
        calls["lane"] += 1
        return _lane_ok(model)

    monkeypatch.setattr(cli, "_invoke_lane", invoke)
    env = _profile_env(tmp_path, REVIEW_PROFILES=DEEP_TWO_LANE)
    assert cli.main(["setup"], env) == 0
    envelope = json.loads((tmp_path / "lanes" / "review-context.json").read_text(encoding="utf-8"))
    digest = envelope["sha256"]
    lane_env = _context_env(tmp_path, envelope, digest, LANE_INDEX="0")
    assert cli.main(["lane"], lane_env) == 0
    lane_env["LANE_INDEX"] = "1"
    assert cli.main(["lane"], {**lane_env, "LANE_MODEL": "vendor/optional"}) == 0
    judge_env = _context_env(tmp_path, envelope, digest)
    artifacts = tmp_path / "downloaded-lanes"
    artifacts.mkdir()
    for index in range(2):
        (tmp_path / "lanes" / f"lane-{index}.json").replace(artifacts / f"lane-{index}.json")
    judge_env["LANE_RESULTS_DIR"] = str(artifacts)
    monkeypatch.setattr(
        cli,
        "_collect_with_loop",
        lambda *a, **k: pytest.fail("judge must not recollect"),
    )
    _install_judge_skip(monkeypatch)
    assert cli.main(["judge"], judge_env) == 0
    assert calls == {"collect": 1, "lane": 2}


def test_all_role_reuses_frozen_context_without_recollection(tmp_path, monkeypatch) -> None:
    calls = {"collect": 0}

    def collect(_env, with_replies=True):
        calls["collect"] += 1
        return _collected(), _loop(), ""

    monkeypatch.setattr(cli, "_collect_with_loop", collect)
    monkeypatch.setattr(cli, "_prepare_workspace", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_maybe_status", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_github", lambda _env: FakeGitHub())
    monkeypatch.setattr(cli, "_invoke_lane", lambda _env, model, *_a, **_k: _lane_ok(model))
    _install_judge_skip(monkeypatch)
    envelope, digest = _run_setup(tmp_path, monkeypatch)
    env = _context_env(tmp_path, envelope, digest)
    assert cli.main(["all"], env) == 0
    assert calls["collect"] == 1


def test_required_lane_missing_keeps_survivor_without_authoritative_ledger(
    tmp_path, monkeypatch
) -> None:
    finding = Finding("Survivor", "body", "bug", "storage/a.py", 1, "vendor/optional")
    _install_collect(monkeypatch)
    monkeypatch.setattr(cli, "_github", lambda _env: FakeGitHub())
    monkeypatch.setattr(
        cli,
        "_invoke_lane",
        lambda _env, model, *_a, **_k: (
            failed_lane("vendor/required", "required lane failed")
            if model == "vendor/required"
            else _lane_ok(model, finding=finding)
        ),
    )
    env = _profile_env(tmp_path, REVIEW_PROFILES=DEEP_TWO_LANE)
    github = FakeGitHub()
    monkeypatch.setattr(cli, "_github", lambda _env: github)
    assert cli.main(["all"], env) == 1
    body = github.posted[0]["body"]
    assert "Survivor" in body
    assert "**Verdict:** `partial`" in body
    assert extract_ledger(body, repo=REPO, pr_number=9) is None
    outputs = (tmp_path / "outputs.txt").read_text(encoding="utf-8")
    assert "profile_satisfied=false" in outputs
    assert "panel_status=required_missing" in outputs


@pytest.mark.parametrize(
    ("optional_ok", "expected_panel", "expected_exit"),
    [
        (True, "degraded", 0),
        (False, "degraded", 0),
    ],
)
def test_optional_lane_outcomes(
    tmp_path, monkeypatch, optional_ok, expected_panel, expected_exit
) -> None:
    _install_collect(monkeypatch)
    github = FakeGitHub()
    monkeypatch.setattr(cli, "_github", lambda _env: github)
    _install_judge_skip(monkeypatch)

    def invoke(_env, model, *_a, **_k):
        if model == "vendor/optional" and not optional_ok:
            return failed_lane(model, "optional lane failed")
        return _lane_ok(model)

    monkeypatch.setattr(cli, "_invoke_lane", invoke)
    env = _profile_env(tmp_path, REVIEW_PROFILES=DEEP_TWO_LANE)
    assert cli.main(["all"], env) == expected_exit
    outputs = (tmp_path / "outputs.txt").read_text(encoding="utf-8")
    assert f"panel_status={expected_panel}" in outputs
    assert "profile_satisfied=true" in outputs


def test_missing_first_lane_keeps_standalone_context_for_judge(tmp_path, monkeypatch) -> None:
    _install_collect(monkeypatch)
    envelope, digest = _run_setup(
        tmp_path, monkeypatch, _profile_env(tmp_path, REVIEW_PROFILES=DEEP_TWO_LANE)
    )
    monkeypatch.setattr(cli, "_github", lambda _env: FakeGitHub())
    lane_env = _context_env(tmp_path, envelope, digest, LANE_INDEX="1")
    monkeypatch.setattr(cli, "_invoke_lane", lambda *_a, **_k: _lane_ok("vendor/optional"))
    assert cli.main(["lane"], lane_env) == 0
    judge_env = _context_env(tmp_path, envelope, digest)
    monkeypatch.setattr(
        cli,
        "_collect_with_loop",
        lambda *a, **k: pytest.fail("judge must not recollect"),
    )
    _install_judge_skip(monkeypatch)
    assert cli.main(["judge"], judge_env) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "digest",
        "lane_index",
        "model",
        "required",
    ],
)
def test_prepared_lane_artifact_rejects_forged_bindings(tmp_path, monkeypatch, mutation) -> None:
    _install_collect(monkeypatch)
    envelope, digest = _run_setup(
        tmp_path, monkeypatch, _profile_env(tmp_path, REVIEW_PROFILES=DEEP_TWO_LANE)
    )
    lane = _lane_ok("vendor/required")
    lane.lane_index = 0
    lane.required = True
    lane.context_sha256 = digest
    lane.head_sha = HEAD_SHA
    lane.review_context = envelope
    payload = lane.to_dict()
    if mutation == "digest":
        payload["context_sha256"] = "f" * 64
    elif mutation == "lane_index":
        payload["lane_index"] = 1
    elif mutation == "model":
        payload["model"] = "vendor/optional"
    else:
        payload["required"] = False
    (tmp_path / "lanes" / "lane-0.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_github", lambda _env: FakeGitHub())
    monkeypatch.setattr(
        cli, "_finish", lambda *a, **k: pytest.fail("must reject before publication")
    )
    with pytest.raises(SchemaError):
        cli._role_judge_prepared(_context_env(tmp_path, envelope, digest))


def test_expired_prepared_workspace_skips_openrouter_before_lane(tmp_path, monkeypatch) -> None:
    _install_collect(monkeypatch)
    envelope, digest = _run_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "_remaining_job_seconds", lambda _env: 0)
    monkeypatch.setattr(
        cli, "require_openrouter_key", lambda _env: pytest.fail("must not read API key")
    )
    context = cli.PreparedContext(restore_context(envelope), digest, envelope)
    result = cli._run_prepared_lane(_context_env(tmp_path, envelope, digest), context, 0)
    assert not result.ok
    assert "deadline expired" in (result.error or "")


def test_deep_initial_and_verify_preserve_continuity(tmp_path, monkeypatch) -> None:
    replies = "Author reply on r1-1"
    verify_loop = _loop(mode="verify")
    verify_generation = verify_loop.generation
    collected = _collected(loop_mode="verify")
    captured: list[str] = []

    def collect(_env, with_replies=True):
        mode = (_env.get("REVIEW_MODE") or "auto").strip().lower()
        if mode == "initial":
            return _collected(loop_mode="initial"), _loop(), ""
        if mode == "auto":
            return _collected(loop_mode="initial"), _loop(), ""
        return collected, verify_loop, replies if with_replies else ""

    monkeypatch.setattr(cli, "_collect_with_loop", collect)
    monkeypatch.setattr(cli, "_prepare_workspace", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_maybe_status", lambda *_a, **_k: None)

    def messages(_env, _collected, _loop, agent_replies=""):
        captured.append(agent_replies)
        return []

    monkeypatch.setattr(cli, "_messages", messages)

    def invoke(_env, model, *_a, **_k):
        return _lane_ok(model)

    monkeypatch.setattr(cli, "_invoke_lane", invoke)
    _install_judge_skip(monkeypatch)

    auto_github = FakeGitHub()
    monkeypatch.setattr(cli, "_github", lambda _env: auto_github)
    assert cli.main(["all"], _profile_env(tmp_path, REVIEW_MODE="auto", REVIEW_LEVEL="deep")) == 0
    auto_ledger = extract_ledger(auto_github.posted[0]["body"], repo=REPO, pr_number=9)
    assert auto_ledger is not None
    assert auto_ledger.generation

    verify_github = FakeGitHub(replies=[("r1-1", replies)])
    monkeypatch.setattr(cli, "_github", lambda _env: verify_github)
    env = _profile_env(tmp_path, REVIEW_MODE="verify", REVIEW_LEVEL="deep")
    assert cli.main(["all"], env) == 0
    assert replies in captured[-1]
    verify_ledger = extract_ledger(verify_github.posted[0]["body"], repo=REPO, pr_number=9)
    assert verify_ledger is not None
    assert verify_ledger.generation == verify_generation

    initial_github = FakeGitHub()
    monkeypatch.setattr(cli, "_github", lambda _env: initial_github)
    assert (
        cli.main(
            ["all"],
            _profile_env(tmp_path, REVIEW_MODE="initial", REVIEW_LEVEL="deep"),
        )
        == 0
    )
    initial_ledger = extract_ledger(initial_github.posted[0]["body"], repo=REPO, pr_number=9)
    assert initial_ledger is not None
    assert initial_ledger.generation
    assert initial_ledger.generation != verify_generation


def test_frozen_runtime_env_overrides_mutable_inputs(tmp_path, monkeypatch) -> None:
    _install_collect(monkeypatch)
    envelope, digest = _run_setup(tmp_path, monkeypatch)
    context = cli.PreparedContext(restore_context(envelope), digest, envelope)
    mutated = dict(
        _context_env(tmp_path, envelope, digest, MODELS="vendor/tampered", EFFORT="high")
    )
    frozen = cli._frozen_execution_env(mutated, context.context)
    assert "vendor/tampered" not in frozen["MODELS"]
    assert frozen["MODELS"] == "vendor/required,vendor/deep"


def test_single_lane_collector_honors_deadline(tmp_path) -> None:
    def run_one(_index: int) -> LaneResult:
        time.sleep(0.2)
        return _lane_ok("vendor/only")

    plan = ReviewPlan(
        "code",
        "deep",
        "manual",
        (ReviewLane("vendor/only", True),),
        "vendor/judge",
        "",
        50,
        60,
        300,
        "a" * 64,
        False,
    )
    context = cli.PreparedContext(
        SimpleNamespace(
            collected=SimpleNamespace(head_sha=HEAD_SHA),
            execution=SimpleNamespace(plan=plan),
        ),
        "c" * 64,
        {"version": 3},
    )
    lanes = cli._collect_all_prepared_lanes(plan, context, tmp_path, run_one, 0.05)
    assert len(lanes) == 1
    assert not lanes[0].ok
    assert "deadline reached" in (lanes[0].error or "")


def test_prepared_lane_clamps_to_collector_cap(tmp_path, monkeypatch) -> None:
    _install_collect(monkeypatch)
    envelope, digest = _run_setup(tmp_path, monkeypatch)
    context = cli.PreparedContext(restore_context(envelope), digest, envelope)
    captured: list[float] = []
    monkeypatch.setattr(
        cli,
        "_invoke_lane",
        lambda *_a, **_k: captured.append(_k["lane_timeout"]) or _lane_ok("vendor/required"),
    )
    monkeypatch.setattr(cli, "_remaining_job_seconds", lambda _env: 600)
    cli._run_prepared_lane(
        _context_env(tmp_path, envelope, digest),
        context,
        0,
        collector_deadline_monotonic=time.monotonic() + 10,
    )
    assert captured and captured[0] <= 5


def test_prepared_setup_one_lane_emits_publisher_judge_needed(tmp_path, monkeypatch) -> None:
    _install_collect(monkeypatch)
    env = _profile_env(tmp_path, REVIEW_LEVEL="auto")
    assert cli.main(["setup"], env) == 0
    outputs = (tmp_path / "outputs.txt").read_text(encoding="utf-8")
    assert "judge_needed=true" in outputs
    envelope = json.loads((tmp_path / "lanes" / "review-context.json").read_text(encoding="utf-8"))
    restored = restore_context(envelope)
    assert cli._prepared_judge_needed(restored) is False


def test_prepared_single_lane_matrix_publishes_without_llm_judge(tmp_path, monkeypatch) -> None:
    calls = {"collect": 0, "lane": 0, "judge": 0}

    def collect(_env, with_replies=True):
        calls["collect"] += 1
        return _collected(), _loop(), ""

    monkeypatch.setattr(cli, "_collect_with_loop", collect)
    monkeypatch.setattr(cli, "_prepare_workspace", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_maybe_status", lambda *_a, **_k: None)
    github = FakeGitHub()
    monkeypatch.setattr(cli, "_github", lambda _env: github)

    def invoke(_env, model, *_a, **_k):
        calls["lane"] += 1
        return _lane_ok(model)

    monkeypatch.setattr(cli, "_invoke_lane", invoke)
    monkeypatch.setattr(
        cli,
        "run_llm_judge",
        lambda *_a, **_k: pytest.fail("single-lane prepared matrix must not call LLM judge"),
    )

    env = _profile_env(tmp_path, REVIEW_LEVEL="auto")
    assert cli.main(["setup"], env) == 0
    outputs = (tmp_path / "outputs.txt").read_text(encoding="utf-8")
    assert "judge_needed=true" in outputs
    envelope = json.loads((tmp_path / "lanes" / "review-context.json").read_text(encoding="utf-8"))
    digest = envelope["sha256"]
    lane_env = _context_env(tmp_path, envelope, digest, LANE_INDEX="0")
    assert cli.main(["lane"], lane_env) == 0
    assert not github.posted
    judge_env = _context_env(tmp_path, envelope, digest)
    artifacts = tmp_path / "downloaded-lanes"
    artifacts.mkdir()
    (tmp_path / "lanes" / "lane-0.json").replace(artifacts / "lane-0.json")
    monkeypatch.setattr(
        cli,
        "_collect_with_loop",
        lambda *a, **k: pytest.fail("judge must not recollect"),
    )
    judge_env["LANE_RESULTS_DIR"] = str(artifacts)
    assert cli.main(["judge"], judge_env) == 0
    assert github.posted
    assert calls == {"collect": 1, "lane": 1, "judge": 0}


def test_prepared_single_lane_required_missing_publishes_error(tmp_path, monkeypatch) -> None:
    _install_collect(monkeypatch)
    github = FakeGitHub()
    monkeypatch.setattr(cli, "_github", lambda _env: github)
    monkeypatch.setattr(
        cli,
        "_invoke_lane",
        lambda _env, model, *_a, **_k: failed_lane(model, "required lane failed"),
    )
    env = _profile_env(tmp_path, REVIEW_LEVEL="auto")
    assert cli.main(["all"], env) == 1
    assert github.posted
    outputs = (tmp_path / "outputs.txt").read_text(encoding="utf-8")
    assert "panel_status=required_missing" in outputs


def test_prepared_matrix_required_missing_lane_publishes_via_judge(tmp_path, monkeypatch) -> None:
    _install_collect(monkeypatch)
    github = FakeGitHub()
    monkeypatch.setattr(cli, "_github", lambda _env: github)
    monkeypatch.setattr(cli, "_maybe_status", lambda *_a, **_k: None)
    monkeypatch.setattr(
        cli,
        "_invoke_lane",
        lambda _env, model, *_a, **_k: failed_lane(model, "required lane failed"),
    )
    env = _profile_env(tmp_path, REVIEW_LEVEL="auto")
    assert cli.main(["setup"], env) == 0
    envelope = json.loads((tmp_path / "lanes" / "review-context.json").read_text(encoding="utf-8"))
    digest = envelope["sha256"]
    lane_env = _context_env(tmp_path, envelope, digest, LANE_INDEX="0")
    assert cli.main(["lane"], lane_env) == 0
    judge_env = _context_env(tmp_path, envelope, digest)
    artifacts = tmp_path / "downloaded-lanes"
    artifacts.mkdir()
    (tmp_path / "lanes" / "lane-0.json").replace(artifacts / "lane-0.json")
    judge_env["LANE_RESULTS_DIR"] = str(artifacts)
    assert cli.main(["judge"], judge_env) == 1
    assert github.posted


def test_prepared_all_posts_reviewing_status_once(tmp_path, monkeypatch) -> None:
    calls = {"collect": 0}
    status_calls: list[tuple[int, str]] = []

    def collect(_env, with_replies=True):
        calls["collect"] += 1
        return _collected(), _loop(), "frozen reply snapshot"

    monkeypatch.setattr(cli, "_collect_with_loop", collect)
    monkeypatch.setattr(cli, "_prepare_workspace", lambda *_a, **_k: None)

    def status(_env, pr_number, body):
        status_calls.append((pr_number, body))

    monkeypatch.setattr(cli, "_maybe_status", status)
    monkeypatch.setattr(cli, "_github", lambda _env: FakeGitHub())
    monkeypatch.setattr(cli, "_invoke_lane", lambda _env, model, *_a, **_k: _lane_ok(model))
    _install_judge_skip(monkeypatch)
    env = _profile_env(tmp_path, STATUS_COMMENTS="true")
    assert cli.main(["all"], env) == 0
    assert len([item for item in status_calls if "Reviewing with OpenRouter" in item[1]]) == 1
    assert calls["collect"] == 1


def test_lane_budget_honors_prepared_lane_ceiling() -> None:
    timeout, _ = cli._lane_budget(
        3600,
        judge_needed=True,
        shares_job_with_judge=True,
        lane_ceiling=1500,
    )
    assert timeout == 1500


def test_lane_budget_shrinks_with_remaining_budget() -> None:
    timeout, _ = cli._lane_budget(
        1200,
        judge_needed=True,
        shares_job_with_judge=True,
        lane_ceiling=1500,
    )
    assert timeout < 1500
    assert timeout >= 1


def test_lane_budget_legacy_default_unchanged() -> None:
    timeout, _ = cli._lane_budget(3600, judge_needed=True, shares_job_with_judge=True)
    assert timeout == DEFAULT_LANE_TIMEOUT_SECONDS


def test_prepared_lane_timeout_uses_profile_ceiling_with_budget(tmp_path, monkeypatch) -> None:
    long_lane_registry = """{
      "version": 1,
      "profiles": {
        "code": {
          "standard": {
            "lanes": [{"model": "vendor/required", "required": true}],
            "lane_timeout_seconds": 1500,
            "job_budget_seconds": 3600
          }
        }
      }
    }"""
    _install_collect(monkeypatch)
    envelope, digest = _run_setup(
        tmp_path,
        monkeypatch,
        _profile_env(
            tmp_path,
            REVIEW_PROFILES=long_lane_registry,
            REVIEW_LEVEL="auto",
            JOB_BUDGET_SECONDS="3600",
            LANE_TIMEOUT_SECONDS="1800",
        ),
    )
    context = cli.PreparedContext(restore_context(envelope), digest, envelope)
    captured: list[float] = []
    monkeypatch.setattr(
        cli,
        "_invoke_lane",
        lambda *_a, **_k: captured.append(_k["lane_timeout"]) or _lane_ok("vendor/required"),
    )
    monkeypatch.setattr(cli, "_remaining_job_seconds", lambda _env: 3600)
    cli._run_prepared_lane(_context_env(tmp_path, envelope, digest), context, 0)
    assert captured == [1500]

    captured.clear()
    monkeypatch.setattr(cli, "_remaining_job_seconds", lambda _env: 1200)
    cli._run_prepared_lane(_context_env(tmp_path, envelope, digest), context, 0)
    assert captured and captured[0] < 1500
