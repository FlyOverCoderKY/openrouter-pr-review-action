"""CLI entry for the composite action roles: setup, lane, judge, all."""

from __future__ import annotations

import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from or_pr_review.collect import (
    CollectedReview,
    collect_review,
    normalize_sha,
    parse_mode,
    parse_scope,
    resolve_mode,
)
from or_pr_review.errors import ActionError, LaneError, SchemaError
from or_pr_review.github_ops import GitHub, upsert_status_comment
from or_pr_review.harness import parse_max_tool_turns, require_openrouter_key, run_lane
from or_pr_review.judge import run_llm_judge
from or_pr_review.merge import MergedIssue, issues_from_single_lane
from or_pr_review.models import (
    LANE_CAP,
    judge_is_needed,
    matrix_json,
    models_json,
    parse_judge_model,
    parse_models,
)
from or_pr_review.prompt import build_messages
from or_pr_review.publish import (
    decide_verdict,
    fail_on_should_fail,
    render_incomplete,
    render_review,
)
from or_pr_review.redaction import redact
from or_pr_review.schema import LaneResult, failed_lane, parse_lane_artifact
from or_pr_review.workspace import materialize_commit

_ACTIVE_ENV: dict[str, str] = {}


def main(argv: list[str] | None = None, env: dict[str, str] | None = None) -> int:
    global _ACTIVE_ENV
    args = list(sys.argv[1:] if argv is None else argv)
    environ = env if env is not None else dict(os.environ)
    _ACTIVE_ENV = environ
    role = (args[0] if args else environ.get("ROLE") or "all").strip().lower()
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
    result, collected = _run_one_lane(env, model)
    path = _write_lane_file(env, index, result)
    _set_output("lane_file", str(path))
    _set_output("lane_ok", "true" if result.ok else "false")
    if result.ok:
        print(f"lane {index} `{model}` ok: {len(result.findings)} finding(s)")
    else:
        print(f"lane {index} `{model}` failed-open: {result.error}")
    if not _judge_needed(env, slugs):
        return _finish(env, [result], collected=collected)
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
    custom = env.get("CUSTOM_INSTRUCTIONS") or ""
    if len(custom.encode("utf-8")) > 16_000:
        raise ActionError("custom_instructions exceeds 16,000 UTF-8 bytes")
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
) -> tuple[list[MergedIssue], str]:
    if not successful:
        return [], "skipped (no successful lanes)"
    if not _judge_needed(env, slugs):
        print("judge skipped: one review lane; posting that lane directly")
        return (
            issues_from_single_lane(successful[0]),
            "skipped (single review lane; one reviewer = no judge)",
        )
    judge_model = parse_judge_model(env.get("JUDGE_MODEL"))
    key = require_openrouter_key(env)
    print(f"judge running with `{judge_model}` (reasoning effort=minimal)")
    issues = run_llm_judge(
        model=judge_model,
        lanes=[lane.to_dict() for lane in lanes],
        api_key=key,
        timeout=_int_env(env, "OPENROUTER_TIMEOUT_SECONDS", 180),
    )
    return issues, f"`{judge_model}`"


def _role_all(env: dict[str, str]) -> int:
    slugs = _validate_inputs(env)
    needed = judge_is_needed(slugs)
    _set_output("models_json", models_json(slugs))
    _set_output("matrix", matrix_json(slugs))
    _set_output("lane_count", str(len(slugs)))
    _set_output("judge_needed", "true" if needed else "false")
    _set_output("judge_model", parse_judge_model(env.get("JUDGE_MODEL")))
    collected = _collect(env)
    _maybe_status(
        env,
        collected.pr_number,
        f"Reviewing with OpenRouter ({len(slugs)} lane(s): {', '.join(f'`{s}`' for s in slugs)}).",
    )
    work = Path(env.get("WORK") or "").resolve() if env.get("WORK") else Path(env.get("RUNNER_TEMP") or "/tmp")
    workspace = _prepare_workspace(env, collected, work)
    messages = _messages(env, collected)

    def _one(model: str) -> LaneResult:
        return _invoke_lane(env, model, messages, workspace)

    lanes: list[LaneResult] = []
    if len(slugs) == 1:
        lanes.append(_one(slugs[0]))
    else:
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
            lanes = [by_index[i] for i in range(len(slugs))]

    for lane in lanes:
        lane.head_sha = collected.head_sha

    lane_dir = work / "lanes"
    lane_dir.mkdir(parents=True, exist_ok=True)
    for index, lane in enumerate(lanes):
        (lane_dir / f"lane-{index}.json").write_text(
            json.dumps(lane.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
    return _finish(env, lanes, collected=collected)


def _run_one_lane(env: dict[str, str], model: str) -> tuple[LaneResult, CollectedReview]:
    collected = _collect(env)
    work = Path(env.get("WORK") or env.get("RUNNER_TEMP") or "/tmp")
    workspace = _prepare_workspace(env, collected, work)
    messages = _messages(env, collected)
    _maybe_status(env, collected.pr_number, f"Lane `{model}` is reviewing via OpenRouter.")
    result = _invoke_lane(env, model, messages, workspace)
    result.head_sha = collected.head_sha
    return result, collected


def _invoke_lane(
    env: dict[str, str],
    model: str,
    messages: list[dict[str, Any]],
    workspace: Path | None,
) -> LaneResult:
    try:
        key = require_openrouter_key(env)
        return run_lane(
            model=model,
            messages=messages,
            api_key=key,
            workspace=workspace,
            max_tool_turns=parse_max_tool_turns(env.get("MAX_TOOL_TURNS")),
            effort=(env.get("EFFORT") or "").strip(),
            timeout=_int_env(env, "OPENROUTER_TIMEOUT_SECONDS", 180),
        )
    except ActionError:
        raise
    except LaneError as exc:
        return failed_lane(model, redact(str(exc)))
    except Exception as exc:  # noqa: BLE001
        return failed_lane(model, redact(str(exc)))


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
    )


def _prepare_workspace(env: dict[str, str], collected: CollectedReview, work: Path) -> Path | None:
    if parse_max_tool_turns(env.get("MAX_TOOL_TURNS")) == 0:
        return None
    source = Path(env.get("SOURCE_WORKSPACE") or env.get("GITHUB_WORKSPACE") or ".").resolve()
    dest = work / "inert-checkout"
    if dest.exists() and any(dest.iterdir()):
        return dest
    try:
        return materialize_commit(source, collected.head_sha, dest)
    except ActionError as exc:
        # Fail closed: the prompt mandates blast-radius tool use, so a
        # silently tool-less run could post an unmarked glance review.
        raise ActionError(
            "inert checkout unavailable; refusing a tool-less review "
            f"(set max_tool_turns: 0 to review without tools): {redact(str(exc))}"
        ) from exc


def _messages(env: dict[str, str], collected: CollectedReview) -> list[dict[str, str]]:
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
    )


def _finish(
    env: dict[str, str],
    lanes: list[LaneResult],
    collected: CollectedReview | None = None,
) -> int:
    if collected is None:
        collected = _collect(env)
    # Lanes stamp the commit they actually reviewed; mixed artifacts are
    # irreconcilable and fail closed before anything posts.
    reviewed_sha = _common_lane_sha(lanes) or collected.head_sha
    successful = [lane for lane in lanes if lane.ok]
    slugs = parse_models(env.get("MODELS"))
    issues, judge_note = _resolve_issues(env, slugs, lanes, successful)
    github = _github(env)
    stale_notice: str | None = None
    live_head = _live_head(github, collected.pr_number)
    if live_head and live_head != reviewed_sha:
        stale_notice = (
            "The PR head advanced after this review's diff was collected. "
            f"This review is pinned to commit {reviewed_sha[:12]} and does not "
            "cover the newest push."
        )
    verdict = decide_verdict(
        issues=issues,
        truncated=collected.truncation.truncated,
        successful_lanes=len(successful),
        fallback=collected.plan.fallback_notice is not None,
        stale=stale_notice is not None,
    )
    body = render_review(
        collected=collected,
        lanes=lanes,
        issues=issues,
        verdict=verdict,
        run_url=env.get("RUN_URL") or "",
        judge_note=judge_note,
        reviewed_sha=reviewed_sha,
        extra_notices=[stale_notice] if stale_notice else None,
    )
    review_url = ""
    try:
        posted = github.create_review(collected.pr_number, body, reviewed_sha)
        html = posted.get("html_url")
        review_url = html if isinstance(html, str) else ""
    except ActionError as exc:
        _maybe_status(
            env,
            collected.pr_number,
            render_incomplete(stage="post-review", reason=redact(str(exc)), run_url=env.get("RUN_URL") or ""),
        )
        raise ActionError(f"failed to post the GitHub review: {exc}") from exc

    _maybe_status(
        env,
        collected.pr_number,
        f"OpenRouter review posted (`{verdict}`). {review_url}".strip(),
    )

    bug_count = sum(1 for issue in issues if issue.severity == "bug")
    _set_output("verdict", verdict)
    _set_output("issue_count", str(len(issues)))
    _set_output("bug_count", str(bug_count))
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
    if fail_on_should_fail(fail_on, issues):
        _error(f"fail_on={fail_on} matched {len(issues)} finding(s) ({bug_count} bug)")
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
    head = pr.get("headRefOid")
    if isinstance(head, str):
        return normalize_sha(head)
    nested = pr.get("head")
    if isinstance(nested, dict) and isinstance(nested.get("sha"), str):
        return normalize_sha(nested["sha"])
    return None


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
                lanes.append(failed_lane(model, "lane artifact missing (job failed or was cancelled)"))
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
