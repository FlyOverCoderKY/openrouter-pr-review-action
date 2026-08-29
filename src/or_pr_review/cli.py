"""CLI entry for the composite action roles: setup, lane, judge, all."""

from __future__ import annotations

import json
import os
import secrets
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from or_pr_review.collect import (
    DIVERGED_NOTICE,
    CollectedReview,
    collect_review,
    head_sha_from_pr,
    parse_mode,
    parse_scope,
    resolve_mode,
)
from or_pr_review.errors import ActionError, LaneError, SchemaError
from or_pr_review.github_ops import GitHub, upsert_status_comment
from or_pr_review.harness import (
    DEFAULT_LANE_TIMEOUT_SECONDS,
    MAX_HTTP_ATTEMPTS,
    MAX_RETRY_AFTER_SECONDS,
    parse_max_tool_turns,
    require_openrouter_key,
    run_lane,
    sanitize_anchors,
)
from or_pr_review.judge import deterministic_union, run_llm_judge
from or_pr_review.loop import (
    Ledger,
    LoopState,
    apply_round,
    apply_severity_floor,
    decide_loop_state,
    encode_ledger,
    latest_ledger,
    merge_resolutions,
    render_agent_context,
    round_report,
)
from or_pr_review.merge import MergedIssue, issues_from_single_lane
from or_pr_review.models import (
    LANE_CAP,
    judge_is_needed,
    matrix_json,
    models_json,
    parse_judge_model,
    parse_models,
)
from or_pr_review.prompt import (
    build_messages,
    changed_paths_from_diff,
    diff_right_side_lines,
    parse_path_profiles,
)
from or_pr_review.publish import (
    decide_verdict,
    fail_on_should_fail,
    inline_review_comments,
    render_incomplete,
    render_review_parts,
)
from or_pr_review.redaction import redact
from or_pr_review.schema import (
    MAX_COVERAGE_ENTRIES,
    LaneResult,
    coverage_count_mismatches,
    failed_lane,
    parse_lane_artifact,
)
from or_pr_review.triage import parse_generated_globs
from or_pr_review.workspace import materialize_commit

_ACTIVE_ENV: dict[str, str] = {}

# The organization workflow gives this action 25 minutes. Stop OpenRouter
# work at 22 minutes so review publication and composite cleanup cannot lose a
# completed lane at the hard GitHub cancellation boundary (the PR358 failure).
JOB_BUDGET_SECONDS = 22 * 60
POST_RESERVE_SECONDS = 3 * 60
JUDGE_SCHEDULING_MARGIN_SECONDS = 5
_JOB_DEADLINE_KEY = "_OR_PR_REVIEW_JOB_DEADLINE_MONOTONIC"


def main(argv: list[str] | None = None, env: dict[str, str] | None = None) -> int:
    global _ACTIVE_ENV
    args = list(sys.argv[1:] if argv is None else argv)
    environ = dict(env) if env is not None else dict(os.environ)
    role = (args[0] if args else environ.get("ROLE") or "all").strip().lower()
    if role in {"all", "lane", "judge"}:
        environ[_JOB_DEADLINE_KEY] = str(time.monotonic() + JOB_BUDGET_SECONDS)
    _ACTIVE_ENV = environ
    try:
        if role == "setup":
            return _role_setup(environ)
        if role == "lane":
            return _role_lane(environ)
        if role == "judge":
            return _role_judge(environ)
        if role == "all":
            return _role_all(environ)
        raise ActionError(f"unknown role {role!r}; expected setup, lane, judge, or all")
    except SchemaError as exc:
        _error(f"schema mismatch (fail-closed): {exc}")
        return 1
    except ActionError as exc:
        _error(redact(str(exc)))
        _best_effort_incomplete(environ, stage=role, reason=redact(str(exc)))
        return 1
    except Exception as exc:  # noqa: BLE001 — unexpected bugs are operational failures
        _error(f"unexpected action error: {redact(str(exc))}")
        traceback.print_exc()
        return 1


def _role_setup(env: dict[str, str]) -> int:
    slugs = _validate_inputs(env)
    needed = judge_is_needed(slugs)
    judge_model = parse_judge_model(env.get("JUDGE_MODEL"))
    _set_output("models_json", models_json(slugs))
    _set_output("matrix", matrix_json(slugs))
    _set_output("lane_count", str(len(slugs)))
    _set_output("judge_needed", "true" if needed else "false")
    _set_output("judge_model", judge_model)
    print(f"parsed {len(slugs)} model lane(s) (cap {LANE_CAP}): {', '.join(slugs)}")
    if needed:
        print(f"judge will run with `{judge_model}`")
    else:
        print("judge skipped: one review lane (one reviewer = no judge)")
    return 0


def _role_lane(env: dict[str, str]) -> int:
    slugs = parse_models(env.get("MODELS"))
    index = _int_env(env, "LANE_INDEX", 0)
    if index < 0:
        raise ActionError(f"LANE_INDEX {index} is invalid")
    override = (env.get("LANE_MODEL") or "").strip()
    if override:
        model = override
    elif index < len(slugs):
        model = slugs[index]
    elif len(slugs) == 1:
        # Reusable workflow matrix jobs pass models=<one slug> plus the global
        # matrix index. Keep that index so lane-N.json artifacts do not collide.
        model = slugs[0]
    else:
        raise ActionError(f"LANE_INDEX {index} is out of range for {len(slugs)} model(s)")
    result, collected, state = _run_one_lane(env, model)
    path = _write_lane_file(env, index, result)
    _set_output("lane_file", str(path))
    _set_output("lane_ok", "true" if result.ok else "false")
    if result.ok:
        print(f"lane {index} `{model}` ok: {len(result.findings)} finding(s)")
    else:
        print(f"lane {index} `{model}` failed-open: {result.error}")
    if not _judge_needed(env, slugs):
        return _finish(env, [result], collected=collected, loop=state)
    return 0


def _role_judge(env: dict[str, str]) -> int:
    expected = parse_models(env.get("MODELS"))
    directory = Path(env.get("LANE_RESULTS_DIR") or "")
    if not directory.is_dir():
        raise ActionError("LANE_RESULTS_DIR is missing or not a directory")
    lanes = _load_lane_dir(directory, expected)
    return _finish(env, lanes)


def _validate_inputs(env: dict[str, str]) -> list[str]:
    slugs = parse_models(env.get("MODELS"))
    parse_judge_model(env.get("JUDGE_MODEL"))
    parse_scope(env.get("REVIEW_SCOPE") or "full-pr")
    parse_mode(env.get("REVIEW_MODE") or "auto")
    fail_on = (env.get("FAIL_ON") or "never").strip().lower()
    if fail_on not in {"never", "bugs", "any"}:
        raise ActionError("fail_on must be never, bugs, or any")
    roast = (env.get("ROAST_LEVEL") or "professional").strip().lower()
    if roast not in {"professional", "playful"}:
        raise ActionError("roast_level must be professional or playful in v1")
    max_diff = _int_env(env, "MAX_DIFF_KB", 300)
    if max_diff <= 0:
        raise ActionError("max_diff_kb must be a positive integer")
    parse_max_tool_turns(env.get("MAX_TOOL_TURNS"))
    _bot_login(env)
    custom = env.get("CUSTOM_INSTRUCTIONS") or ""
    if len(custom.encode("utf-8")) > 16_000:
        raise ActionError("custom_instructions exceeds 16,000 UTF-8 bytes")
    parse_path_profiles(env.get("PATH_PROFILES"))
    parse_generated_globs(env.get("GENERATED_PATHS"))
    return slugs


def _judge_needed(env: dict[str, str], slugs: list[str] | None = None) -> bool:
    flag = (env.get("JUDGE_NEEDED") or "").strip().lower()
    if flag in {"true", "1", "yes"}:
        return True
    if flag in {"false", "0", "no"}:
        return False
    return judge_is_needed(slugs if slugs is not None else parse_models(env.get("MODELS")))


def _resolve_issues(
    env: dict[str, str],
    slugs: list[str],
    lanes: list[LaneResult],
    successful: list[LaneResult],
) -> tuple[list[MergedIssue], str, float | None, bool]:
    if not successful:
        return [], "skipped (no successful lanes)", None, False
    if len(successful) == 1:
        print("judge skipped: one successful lane; posting that lane directly")
        reason = (
            "skipped (single review lane; one reviewer = no judge)"
            if len(lanes) == 1
            else "skipped (one successful review lane; no merge needed)"
        )
        return (
            issues_from_single_lane(successful[0]),
            reason,
            None,
            False,
        )
    if not _judge_needed(env, slugs):
        # Defensive only: model validation currently makes multiple lanes
        # imply a judge unless the caller explicitly overrides judge_needed.
        return (
            deterministic_union([lane.to_dict() for lane in successful]),
            "skipped by configuration (deterministic union)",
            None,
            False,
        )
    judge_model = parse_judge_model(env.get("JUDGE_MODEL"))
    lane_payloads = [lane.to_dict() for lane in lanes]
    judge_timeout = _judge_request_timeout(env)
    if judge_timeout is None:
        print(
            "judge skipped near the job deadline; using the deterministic "
            "recall-safe union",
            flush=True,
        )
        return (
            deterministic_union(lane_payloads),
            f"`{judge_model}` (deadline fallback: deterministic union)",
            None,
            False,
        )
    key = require_openrouter_key(env)
    print(
        f"judge running with `{judge_model}` (reasoning effort=minimal, "
        f"request timeout={judge_timeout}s)",
        flush=True,
    )
    try:
        issues, mode, judge_cost = run_llm_judge(
            model=judge_model,
            lanes=lane_payloads,
            api_key=key,
            timeout=judge_timeout,
        )
    except SchemaError as exc:
        # The lane artifacts already passed our schema and anchor gates. A
        # malformed judge answer cannot erase that validated recall; make the
        # degraded merge explicit on the review instead.
        print(
            f"warning: judge schema failed ({redact(str(exc))}); using the "
            "deterministic recall-safe union",
            flush=True,
        )
        return (
            deterministic_union(lane_payloads),
            f"`{judge_model}` (schema fallback: deterministic union)",
            None,
            False,
        )
    except ActionError as exc:
        # Review lanes are the source of recall; a judge transport failure
        # must not erase their completed work. Schema violations remain
        # fail-closed above because they indicate an internal contract bug.
        print(
            f"warning: judge transport failed ({redact(str(exc))}); using the "
            "deterministic recall-safe union",
            flush=True,
        )
        return (
            deterministic_union(lane_payloads),
            f"`{judge_model}` (transport fallback: deterministic union)",
            None,
            False,
        )
    # Recall-safety outcomes are visible on the posted review, not only in
    # the job log: readers must be able to tell a clean merge from a
    # repaired or fallback (chattier, exact-dedup union) post.
    if mode == "merged":
        return issues, f"`{judge_model}`", judge_cost, True
    return issues, f"`{judge_model}` ({mode}: recall-safe coverage enforced)", judge_cost, True


def _role_all(env: dict[str, str]) -> int:
    slugs = _validate_inputs(env)
    needed = judge_is_needed(slugs)
    _set_output("models_json", models_json(slugs))
    _set_output("matrix", matrix_json(slugs))
    _set_output("lane_count", str(len(slugs)))
    _set_output("judge_needed", "true" if needed else "false")
    _set_output("judge_model", parse_judge_model(env.get("JUDGE_MODEL")))
    collected, state, agent_replies = _collect_with_loop(env)
    _maybe_status(
        env,
        collected.pr_number,
        f"Reviewing with OpenRouter ({len(slugs)} lane(s): {', '.join(f'`{s}`' for s in slugs)}).",
    )
    work = (
        Path(env.get("WORK") or "").resolve()
        if env.get("WORK")
        else Path(env.get("RUNNER_TEMP") or "/tmp")
    )
    workspace = _prepare_workspace(env, collected, work)
    messages = _messages(env, collected, state, agent_replies)
    expect_coverage, expected_paths = _coverage_expectations(state, collected)
    expected_ids = (
        {finding.id for finding in state.open_prior} if state.mode == "verify" else None
    )
    remaining = _remaining_job_seconds(env)
    lane_timeout = DEFAULT_LANE_TIMEOUT_SECONDS
    if remaining is not None:
        lane_timeout = max(
            1,
            min(DEFAULT_LANE_TIMEOUT_SECONDS, int(remaining - POST_RESERVE_SECONDS)),
        )
    lane_dir = work / "lanes"
    lane_dir.mkdir(parents=True, exist_ok=True)

    def _one(model: str) -> LaneResult:
        return _invoke_lane(
            env,
            model,
            messages,
            workspace,
            expect_coverage=expect_coverage,
            expect_resolutions=state.mode == "verify",
            expected_paths=expected_paths,
            expected_resolution_ids=expected_ids,
            lane_timeout=lane_timeout,
        )

    lanes: list[LaneResult] = []
    if len(slugs) == 1:
        lane = _one(slugs[0])
        lane.head_sha = collected.head_sha
        _persist_lane_artifact(lane_dir, 0, lane)
        lanes.append(lane)
    else:
        # Thread workers are safe to join because run_lane bounds every HTTP
        # request and executes each repository tool in a killable subprocess.
        # Do not use shutdown(wait=False): running Python threads cannot be
        # cancelled, and pretending otherwise recreates the PR358 failure.
        with ThreadPoolExecutor(max_workers=min(len(slugs), LANE_CAP)) as pool:
            futures = {pool.submit(_one, model): i for i, model in enumerate(slugs)}
            by_index: dict[int, LaneResult] = {}
            for future in as_completed(futures):
                i = futures[future]
                model = slugs[i]
                try:
                    by_index[i] = future.result()
                except Exception as exc:  # noqa: BLE001
                    by_index[i] = failed_lane(model, redact(str(exc)))
                by_index[i].head_sha = collected.head_sha
                _persist_lane_artifact(lane_dir, i, by_index[i])
                print(
                    f"lane {i} `{model}` persisted "
                    f"({'ok' if by_index[i].ok else 'failed-open'})",
                    flush=True,
                )
        lanes = [by_index[i] for i in range(len(slugs))]
    return _finish(env, lanes, collected=collected, loop=state)


def _run_one_lane(
    env: dict[str, str], model: str
) -> tuple[LaneResult, CollectedReview, LoopState]:
    collected, state, agent_replies = _collect_with_loop(env)
    work = Path(env.get("WORK") or env.get("RUNNER_TEMP") or "/tmp")
    workspace = _prepare_workspace(env, collected, work)
    messages = _messages(env, collected, state, agent_replies)
    _maybe_status(env, collected.pr_number, f"Lane `{model}` is reviewing via OpenRouter.")
    expect_coverage, expected_paths = _coverage_expectations(state, collected)
    remaining = _remaining_job_seconds(env)
    lane_timeout = DEFAULT_LANE_TIMEOUT_SECONDS
    if remaining is not None:
        reserve = (
            0
            if _judge_needed(env, parse_models(env.get("MODELS")))
            else POST_RESERVE_SECONDS
        )
        lane_timeout = max(
            1,
            min(DEFAULT_LANE_TIMEOUT_SECONDS, int(remaining - reserve)),
        )
    result = _invoke_lane(
        env,
        model,
        messages,
        workspace,
        expect_coverage=expect_coverage,
        expect_resolutions=state.mode == "verify",
        expected_paths=expected_paths,
        expected_resolution_ids=(
            {finding.id for finding in state.open_prior}
            if state.mode == "verify"
            else None
        ),
        lane_timeout=lane_timeout,
    )
    result.head_sha = collected.head_sha
    return result, collected, state


def _invoke_lane(
    env: dict[str, str],
    model: str,
    messages: list[dict[str, Any]],
    workspace: Path | None,
    *,
    expect_coverage: bool = False,
    expect_resolutions: bool = False,
    expected_paths: set[str] | None = None,
    expected_resolution_ids: set[str] | None = None,
    lane_timeout: int | None = None,
) -> LaneResult:
    try:
        key = require_openrouter_key(env)
        lane_kwargs: dict[str, Any] = {}
        if lane_timeout is not None:
            lane_kwargs["lane_timeout"] = lane_timeout
        return run_lane(
            model=model,
            messages=messages,
            api_key=key,
            workspace=workspace,
            max_tool_turns=parse_max_tool_turns(env.get("MAX_TOOL_TURNS")),
            effort=(env.get("EFFORT") or "").strip(),
            timeout=_int_env(env, "OPENROUTER_TIMEOUT_SECONDS", 180),
            # Tool-less runs have no inert checkout; the anchor gate then
            # checks against the workflow's own full checkout of the head.
            anchor_root=_source_root(env) if workspace is None else None,
            expect_coverage=expect_coverage,
            expect_resolutions=expect_resolutions,
            expected_paths=expected_paths,
            expected_resolution_ids=expected_resolution_ids,
            **lane_kwargs,
        )
    except ActionError:
        raise
    except LaneError as exc:
        return failed_lane(model, redact(str(exc)))
    except Exception as exc:  # noqa: BLE001
        return failed_lane(model, redact(str(exc)))


def _new_generation() -> str:
    """A fresh nonce per initial round.

    Deriving the token from the reviewed SHA would reuse it when a loop is
    reset at the same commit, letting old inline threads pair with new
    same-numbered findings.
    """
    return secrets.token_hex(6)


def _coverage_expectations(
    state: LoopState, collected: CollectedReview
) -> tuple[bool, set[str] | None]:
    """Whether this run enforces the coverage manifest, and for which paths.

    A diff naming more paths than a manifest may hold would make every lane
    unsatisfiable (the prompt demands every file while the parser caps the
    array), so enforcement degrades to unenforced with a visible notice.
    """
    if state.mode != "initial":
        return False, None
    paths = set(changed_paths_from_diff(collected.diff))
    if len(paths) > MAX_COVERAGE_ENTRIES:
        print(
            f"notice: {len(paths)} diff paths exceed the coverage manifest cap "
            f"({MAX_COVERAGE_ENTRIES}); coverage enforcement is skipped for this run"
        )
        return False, None
    return True, paths


def _bot_login(env: dict[str, str]) -> str:
    login = (env.get("BOT_LOGIN") or "").strip() or "github-actions[bot]"
    if len(login) > 100 or any(character.isspace() for character in login):
        raise ActionError("bot_login must be a GitHub login of at most 100 characters")
    return login


def _resolve_loop(
    env: dict[str, str], github: GitHub, pr_number: int
) -> tuple[Ledger | None, LoopState]:
    """Recover the loop position before collecting the diff.

    State recovery fails closed: a corrupted newest ledger raises instead of
    silently resetting to round 1, and a state-free synchronize run under
    latest-commit scope is refused (an "initial" review of one push could
    report clean without ever seeing the rest of the PR).
    """
    mode_input = parse_mode(env.get("REVIEW_MODE") or "auto")
    event_action = (env.get("EVENT_ACTION") or "").strip().lower()
    scope = parse_scope(env.get("REVIEW_SCOPE") or "full-pr")
    repo = (env.get("GITHUB_REPOSITORY") or "").strip()
    ledger: Ledger | None = None
    if mode_input == "verify" or (mode_input == "auto" and event_action == "synchronize"):
        bodies = github.list_bot_review_bodies(pr_number, _bot_login(env))
        ledger = latest_ledger(bodies, repo=repo, pr_number=pr_number)
        if ledger is None and mode_input == "verify":
            raise ActionError(
                "review_mode is verify but no prior review-loop state exists on "
                "this PR; run an initial review first"
            )
        if ledger is None and event_action == "synchronize" and scope == "latest-commit":
            raise ActionError(
                "this synchronize run collects only the latest commit and no prior "
                "review-loop state exists, so carried findings cannot be verified; "
                "run an initial full-PR review first (review_mode: initial, "
                "review_scope: full-pr)"
            )
    mode, round_number = decide_loop_state(
        review_mode=mode_input, event_action=event_action, ledger=ledger
    )
    prior = ledger.findings if ledger is not None and mode == "verify" else ()
    generation = ledger.generation if ledger is not None and mode == "verify" else ""
    prior, retired = apply_severity_floor(prior, round_number if mode == "verify" else 1)
    return ledger, LoopState(
        mode=mode,
        round_number=round_number,
        prior_findings=prior,
        generation=generation,
        retired_prior=retired,
    )


def _collect_with_loop(
    env: dict[str, str], *, with_replies: bool = True
) -> tuple[CollectedReview, LoopState, str]:
    pr_number = _pr_number(env)
    github = _github(env)
    ledger, state = _resolve_loop(env, github, pr_number)
    env_for_collect = dict(env)
    env_for_collect["REVIEW_MODE"] = state.mode
    if state.mode == "verify" and ledger is not None and ledger.reviewed_sha:
        # Continuity: verify everything since the last successfully published
        # review, not just this push's webhook range, so a run racing a
        # cancelled older run can never skip the commits that run covered.
        env_for_collect["EVENT_BEFORE"] = ledger.reviewed_sha
    collected = _collect(env_for_collect)
    if (
        state.mode == "verify"
        and ledger is not None
        and collected.plan.fallback_notice == DIVERGED_NOTICE
    ):
        # History was rewritten (force-push): the last reviewed SHA is no
        # longer an ancestor, so a latest-commit verify can never cover the
        # rewritten work — and its partial verdict would never republish the
        # ledger, repeating the identical failed round forever. A rewrite
        # requires a fresh exhaustive pass: reset to a full-PR initial round.
        # Transient compare failures (timeouts, 5xx) carry a different notice
        # and never reset: they stay a single-commit partial round and retry
        # naturally on the next push.
        print(
            "notice: history diverged from the last reviewed commit "
            f"({ledger.reviewed_sha[:12]}); resetting to a full-PR initial review"
        )
        env_reset = dict(env)
        env_reset["REVIEW_MODE"] = "initial"
        env_reset["REVIEW_SCOPE"] = "full-pr"
        collected = _collect(env_reset)
        return collected, LoopState(mode="initial", round_number=1), ""
    agent_replies = ""
    if with_replies and state.mode == "verify":
        try:
            # Replies to findings the severity floor retired would reintroduce
            # the retired context and invite re-adjudication; only threads for
            # carried findings (open or disputed) reach the prompt.
            carried_ids = {finding.id for finding in state.prior_findings}
            agent_replies = render_agent_context(
                [
                    reply
                    for reply in github.list_finding_replies(
                        pr_number, generation=state.generation
                    )
                    if reply[0] in carried_ids
                ],
                github.list_recent_issue_comments(pr_number),
            )
        except ActionError as exc:
            print(
                "warning: could not fetch reviewer replies; verifying from "
                f"commits only: {redact(str(exc))}"
            )
    return collected, state, agent_replies


def _collect(env: dict[str, str]) -> CollectedReview:
    pr_number = _pr_number(env)
    github = _github(env)
    scope = parse_scope(env.get("REVIEW_SCOPE") or "full-pr")
    mode = resolve_mode(parse_mode(env.get("REVIEW_MODE") or "auto"), env.get("EVENT_ACTION"))
    max_diff_kb = _int_env(env, "MAX_DIFF_KB", 300)
    return collect_review(
        pr_number=pr_number,
        scope=scope,
        mode=mode,
        before_sha=env.get("EVENT_BEFORE"),
        after_sha=env.get("EVENT_AFTER"),
        head_sha=env.get("HEAD_SHA"),
        max_diff_kb=max_diff_kb,
        source=github,
        gitattributes_text=_gitattributes_text(env),
        generated_globs=parse_generated_globs(env.get("GENERATED_PATHS")),
    )


def _gitattributes_text(env: dict[str, str]) -> str:
    """Best-effort .gitattributes AT THE REVIEWED COMMIT, for triage.

    `git show <head>:.gitattributes` against the workflow checkout's object
    store pins the read to the reviewed commit and fails soft (empty string,
    heuristics-only packing) in any checkout that does not contain that
    commit — e.g. the reusable workflow's judge job, which checks out only
    this action's repository and must not parse a foreign .gitattributes.

    Trust note: .gitattributes is repository content, so on a PR it is
    contributor-controlled. Honoring linguist-generated is still strictly
    safer than the pre-triage behavior it replaces: a demoted file keeps its
    stub, its coverage obligation, and tool access, while under the raw byte
    cut an attacker could push hand-written code beyond the cutoff entirely
    (no stub, no coverage, permanent partial). Demotion can never remove a
    file from review.
    """
    import subprocess

    root = _source_root(env)
    head = (env.get("HEAD_SHA") or env.get("EVENT_AFTER") or "").strip()
    if root is None or not head:
        return ""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "show", f"{head}:.gitattributes"],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    return (proc.stdout or b"").decode("utf-8", errors="replace")


def _source_root(env: dict[str, str]) -> Path | None:
    """The workflow's own checkout, for anchor checks only (never tool use)."""
    raw = (env.get("SOURCE_WORKSPACE") or env.get("GITHUB_WORKSPACE") or "").strip()
    if not raw:
        return None
    root = Path(raw).resolve()
    return root if root.is_dir() else None


def _prepare_workspace(env: dict[str, str], collected: CollectedReview, work: Path) -> Path | None:
    if parse_max_tool_turns(env.get("MAX_TOOL_TURNS")) == 0:
        return None
    source = Path(env.get("SOURCE_WORKSPACE") or env.get("GITHUB_WORKSPACE") or ".").resolve()
    dest = work / "inert-checkout"
    if dest.exists() and any(dest.iterdir()):
        return dest
    try:
        # Stubbed files carry a tool-readability contract, so they may exceed
        # the normal materialization cap (bounded by the oversized ceiling).
        return materialize_commit(
            source,
            collected.head_sha,
            dest,
            oversized_ok=frozenset(collected.truncation.stubbed_files),
        )
    except ActionError as exc:
        # Fail closed: the prompt mandates blast-radius tool use, so a
        # silently tool-less run could post an unmarked glance review.
        raise ActionError(
            "inert checkout unavailable; refusing a tool-less review "
            f"(set max_tool_turns: 0 to review without tools): {redact(str(exc))}"
        ) from exc


def _messages(
    env: dict[str, str],
    collected: CollectedReview,
    state: LoopState | None = None,
    agent_replies: str = "",
) -> list[dict[str, str]]:
    custom = env.get("CUSTOM_INSTRUCTIONS") or ""
    if len(custom.encode("utf-8")) > 16_000:
        raise ActionError("custom_instructions exceeds 16,000 UTF-8 bytes")
    tone = (env.get("ROAST_LEVEL") or "professional").strip().lower()
    if tone not in {"professional", "playful"}:
        raise ActionError("roast_level must be professional or playful in v1")
    # persona is reserved and unused; passed through so a later release can read it.
    return build_messages(
        collected,
        custom_instructions=custom,
        tone=tone,
        persona=env.get("PERSONA") or "",
        loop=state,
        agent_replies=agent_replies,
        path_profiles=parse_path_profiles(env.get("PATH_PROFILES")),
    )


def _finish(
    env: dict[str, str],
    lanes: list[LaneResult],
    collected: CollectedReview | None = None,
    loop: LoopState | None = None,
) -> int:
    if collected is None or loop is None:
        collected, loop, _replies = _collect_with_loop(env, with_replies=False)
    # Lanes stamp the commit they actually reviewed; mixed artifacts are
    # irreconcilable and fail closed before anything posts.
    reviewed_sha = _common_lane_sha(lanes) or collected.head_sha
    successful = [lane for lane in lanes if lane.ok]
    slugs = parse_models(env.get("MODELS"))
    issues, judge_note, judge_cost, judge_ran = _resolve_issues(env, slugs, lanes, successful)
    # Judge output bypasses the per-lane anchor gate, so gate the merged
    # issues too when a checkout of the reviewed head is available (the
    # judge job checks out the same head ref). MergedIssue duck-types the
    # file/line/title fields the gate touches; single-lane issues were
    # already gated in run_lane and pass through unchanged.
    finish_root = _source_root(env)
    if finish_root is not None:
        issues = sanitize_anchors(issues, finish_root)  # type: ignore[arg-type]

    prior_ids = {finding.id for finding in loop.open_prior}
    resolutions = merge_resolutions([lane.resolutions for lane in successful], prior_ids)
    outcome = apply_round(loop, issues, resolutions)
    issues = outcome.issues

    github = _github(env)
    stale_notice: str | None = None
    live_head = _live_head(github, collected.pr_number)
    if live_head and live_head != reviewed_sha:
        stale_notice = (
            "The PR head advanced after this review's diff was collected. "
            f"This review is pinned to commit {reviewed_sha[:12]} and does not "
            "cover the newest push."
        )
    notices: list[str] = []
    if stale_notice:
        notices.append(stale_notice)
    if loop.mode == "initial":
        diff_path_set = set(changed_paths_from_diff(collected.diff))
        for lane in lanes:
            if lane.ok and lane.coverage:
                for note in coverage_count_mismatches(
                    lane.findings, lane.coverage, diff_path_set
                ):
                    notices.append(f"`{lane.model}`: {note}")
    # Diff-budget triage: stub-only truncation (every changed file embedded
    # or stubbed) does not force partial — only dropped files or a raw byte
    # cut do, so dense PRs keep verdicts, ledger publication, and loop
    # continuity. The stub contract is that TOOLS sweep the stubbed files,
    # so a tool-less run cannot honor it: stubs + max_tool_turns=0 stays a
    # partial review.
    truncation_partial = collected.truncation.forces_partial
    if collected.truncation.stubbed_files and parse_max_tool_turns(env.get("MAX_TOOL_TURNS")) == 0:
        truncation_partial = True
        notices.append(
            "Diff-budget triage stubbed "
            f"{len(collected.truncation.stubbed_files)} file(s) but tools are "
            "disabled (max_tool_turns: 0), so the stubbed files could not be "
            "swept. This review is partial."
        )
    verdict = decide_verdict(
        issues=issues,
        truncated=truncation_partial,
        successful_lanes=len(successful),
        fallback=collected.plan.fallback_notice is not None,
        stale=stale_notice is not None,
    )
    if verdict == "clean" and outcome.open_issue_count:
        # Carried findings from earlier rounds are still open.
        verdict = "issues"

    # The generation token scopes inline finding markers to this loop
    # generation; a reset mints a new one so old threads can never pair with
    # new same-numbered findings.
    generation = (
        loop.generation
        if loop.mode == "verify" and loop.generation
        else _new_generation()
    )
    hidden_marker: str | None = None
    if verdict in {"clean", "issues"}:
        # A partial or error run never publishes authoritative loop state;
        # the previous marker remains the retry boundary, so a truncated or
        # fallback diff cannot permanently skip unseen code.
        hidden_marker = encode_ledger(
            replace(outcome.ledger, reviewed_sha=reviewed_sha, generation=generation),
            repo=(env.get("GITHUB_REPOSITORY") or "").strip(),
            pr_number=collected.pr_number,
        )

    bodies = render_review_parts(
        collected=collected,
        lanes=lanes,
        issues=issues,
        verdict=verdict,
        run_url=env.get("RUN_URL") or "",
        judge_note=judge_note,
        judge_cost=judge_cost,
        judge_ran=judge_ran,
        reviewed_sha=reviewed_sha,
        extra_notices=notices or None,
        hidden_marker=hidden_marker,
        round_lines=round_report(loop, outcome) or None,
    )
    comments: list[dict[str, Any]] = []
    if verdict in {"clean", "issues"}:
        # Partial runs never post inline comments: their diff (stale head,
        # truncated, or fallback) is not what the anchors were computed
        # against, and a same-round retry would re-issue the same finding ids.
        comments = inline_review_comments(
            issues,
            allowed_lines=diff_right_side_lines(collected.diff),
            generation=generation,
        )
    review_url = ""
    try:
        if comments:
            posted = github.create_review(
                collected.pr_number, bodies[0], reviewed_sha, comments=comments
            )
        else:
            posted = github.create_review(collected.pr_number, bodies[0], reviewed_sha)
        html = posted.get("html_url")
        review_url = html if isinstance(html, str) else ""
    except ActionError as exc:
        _maybe_status(
            env,
            collected.pr_number,
            render_incomplete(
                stage="post-review",
                reason=redact(str(exc)),
                run_url=env.get("RUN_URL") or "",
            ),
        )
        raise ActionError(f"failed to post the GitHub review: {exc}") from exc

    for continuation in bodies[1:]:
        try:
            github.create_issue_comment(collected.pr_number, continuation)
        except ActionError as exc:
            print(f"warning: could not post a continuation comment: {redact(str(exc))}")

    _maybe_status(
        env,
        collected.pr_number,
        f"OpenRouter review posted (`{verdict}`). {review_url}".strip(),
    )

    _set_output("verdict", verdict)
    _set_output("issue_count", str(outcome.open_issue_count))
    _set_output("bug_count", str(outcome.open_bug_count))
    _set_output("round", str(loop.round_number))
    _set_output("review_url", review_url)

    _set_output("judge_needed", "true" if _judge_needed(env, slugs) else "false")
    _set_output("judge_model", parse_judge_model(env.get("JUDGE_MODEL")))

    if verdict == "error":
        raise ActionError(
            "every model lane failed; nothing structured arrived to post"
        )

    fail_on = (env.get("FAIL_ON") or "never").strip().lower()
    if fail_on not in {"never", "bugs", "any"}:
        raise ActionError("fail_on must be never, bugs, or any")
    if fail_on_should_fail(
        fail_on,
        issues,
        open_issue_count=outcome.open_issue_count,
        open_bug_count=outcome.open_bug_count,
    ):
        _error(
            f"fail_on={fail_on} matched {outcome.open_issue_count} open finding(s) "
            f"({outcome.open_bug_count} bug)"
        )
        return 1
    return 0


def _common_lane_sha(lanes: list[LaneResult]) -> str | None:
    """The one commit every lane reviewed, or None when no lane recorded one.

    Mixed artifacts (lanes that reviewed different commits) fail closed:
    merging findings from two different code states would attribute results
    to a commit no lane actually reviewed.
    """
    shas = {lane.head_sha for lane in lanes if lane.head_sha}
    if len(shas) > 1:
        listed = ", ".join(sorted(sha[:12] for sha in shas))
        raise SchemaError(
            f"lane artifacts reviewed different commits ({listed}); refusing to merge them"
        )
    return next(iter(shas), None)


def _live_head(github: GitHub, pr_number: int) -> str | None:
    try:
        pr = github.pr_view(pr_number)
    except ActionError as exc:
        print(f"warning: could not re-check the live PR head: {redact(str(exc))}")
        return None
    return head_sha_from_pr(pr)


def _load_lane_dir(directory: Path, expected: list[str]) -> list[LaneResult]:
    files = sorted(directory.rglob("lane-*.json"))
    by_index: dict[int, Path] = {}
    for path in files:
        stem = path.stem  # lane-0
        if not stem.startswith("lane-"):
            continue
        suffix = stem.split("-", 1)[1]
        if suffix.isdigit():
            by_index[int(suffix)] = path

    lanes: list[LaneResult] = []
    for index, model in enumerate(expected):
        path = by_index.get(index)
        if path is None:
            # also accept a lone file for a 1-lane job
            if len(expected) == 1 and files:
                path = files[0]
            else:
                lanes.append(
                    failed_lane(model, "lane artifact missing (job failed or was cancelled)")
                )
                continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SchemaError(f"{path.name} is not valid JSON: {exc}") from exc
        artifact = parse_lane_artifact(payload)
        lanes.append(artifact)
    return lanes


def _write_lane_file(env: dict[str, str], index: int, result: LaneResult) -> Path:
    # Persist outside the action's mktemp WORK dir so upload-artifact can
    # still see the file after the composite cleanup step.
    explicit = (env.get("LANE_RESULTS_DIR") or "").strip()
    if explicit:
        directory = Path(explicit)
    else:
        directory = Path(env.get("RUNNER_TEMP") or "/tmp") / "or-pr-review-lanes"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"lane-{index}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def _persist_lane_artifact(directory: Path, index: int, result: LaneResult) -> Path:
    """Persist an all-role lane immediately, before waiting for siblings."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"lane-{index}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def _remaining_job_seconds(env: dict[str, str]) -> float | None:
    raw = (env.get(_JOB_DEADLINE_KEY) or "").strip()
    if not raw:
        return None
    try:
        deadline = float(raw)
    except ValueError:
        return None
    return max(0.0, deadline - time.monotonic())


def _judge_request_timeout(env: dict[str, str]) -> int | None:
    """Fit every possible judge retry inside the remaining job budget."""
    configured = _int_env(env, "OPENROUTER_TIMEOUT_SECONDS", 180)
    remaining = _remaining_job_seconds(env)
    if remaining is None:
        return configured
    judge_budget = remaining - POST_RESERVE_SECONDS
    retry_reserve = (MAX_HTTP_ATTEMPTS - 1) * MAX_RETRY_AFTER_SECONDS
    usable = judge_budget - retry_reserve - JUDGE_SCHEDULING_MARGIN_SECONDS
    if usable < MAX_HTTP_ATTEMPTS:
        return None
    return min(configured, max(1, int(usable // MAX_HTTP_ATTEMPTS)))


def _github(env: dict[str, str]) -> GitHub:
    token = (env.get("GITHUB_TOKEN") or env.get("GH_TOKEN") or "").strip()
    repository = (env.get("GITHUB_REPOSITORY") or "").strip()
    timeout = _int_env(env, "GITHUB_TIMEOUT_SECONDS", 120)
    return GitHub(token=token, repository=repository, timeout=timeout)


def _pr_number(env: dict[str, str]) -> int:
    raw = (env.get("PR_NUMBER") or "").strip()
    if not raw:
        raise ActionError("pr_number is empty (set the input or run on a pull_request event)")
    try:
        number = int(raw)
    except ValueError as exc:
        raise ActionError("pr_number must be an integer") from exc
    if number <= 0:
        raise ActionError("pr_number must be a positive integer")
    return number


def _int_env(env: dict[str, str], name: str, default: int) -> int:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ActionError(f"{name} must be an integer") from exc


def _maybe_status(env: dict[str, str], pr_number: int, body: str) -> None:
    flag = (env.get("STATUS_COMMENTS") or "true").strip().lower()
    if flag not in {"true", "1", "yes"}:
        return
    try:
        upsert_status_comment(_github(env), pr_number=pr_number, body=body, enabled=True)
    except ActionError as exc:
        print(f"warning: status comment failed: {redact(str(exc))}")


def _best_effort_incomplete(env: dict[str, str], *, stage: str, reason: str) -> None:
    try:
        pr_number = _pr_number(env)
    except ActionError:
        return
    _maybe_status(
        env,
        pr_number,
        render_incomplete(stage=stage, reason=reason, run_url=env.get("RUN_URL") or ""),
    )


def _set_output(name: str, value: str) -> None:
    path = _ACTIVE_ENV.get("GITHUB_OUTPUT") or os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        if "\n" in value:
            handle.write(f"{name}<<ORPR_EOF\n{value}\nORPR_EOF\n")
        else:
            handle.write(f"{name}={value}\n")


def _error(message: str) -> None:
    print(f"::error::{redact(message)}", file=sys.stderr)
    print(redact(message), file=sys.stderr)
