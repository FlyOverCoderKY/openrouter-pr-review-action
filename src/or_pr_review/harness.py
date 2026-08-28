"""Thin OpenRouter chat-completions harness with optional read-only tools."""

from __future__ import annotations

import http.client
import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from or_pr_review.errors import ActionError, LaneError
from or_pr_review.redaction import redact
from or_pr_review.schema import (
    SCHEMA_VERSION,
    Finding,
    LaneResult,
    Resolution,
    failed_lane,
    findings_json_schema,
    parse_lane_payload,
    parse_model_findings,
    validate_coverage,
)
from or_pr_review.workspace import READ_ONLY_TOOLS, dispatch_tool

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HTTP_REFERER = "https://github.com/FlyOverCoderKY/openrouter-pr-review-action"
APP_TITLE = "OpenRouter PR Review Action"
DEFAULT_TIMEOUT = 180
# Transient-error policy: each request retries a few times with backoff, and
# a mid-loop failure that survives the retries triggers one salvage attempt
# in _run_loop instead of discarding the gathered evidence.
RETRYABLE_STATUS = (408, 429, 500, 502, 503, 504)
MAX_HTTP_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 2.0
MAX_RETRY_AFTER_SECONDS = 30.0
# First-pass budget matches the sibling Grok action's default max_turns=50.
# Follow-up jobs may pass a lower value (sibling callers often use 30).
DEFAULT_MAX_TOOL_TURNS = 50
MAX_TOOL_TURNS = DEFAULT_MAX_TOOL_TURNS
MAX_TOOL_TURNS_LIMIT = 1000
# Aggregate cap on tool-observation bytes per lane. Every observation is
# resent on every later request, so unbounded reads grow the transcript
# quadratically; past this cap the loop withdraws tools and asks for the
# JSON finish.
MAX_OBSERVATION_BYTES = 600_000
BLAST_RADIUS_NUDGE = (
    "You returned a review without using tools. The embedded diff is not the "
    "whole repository. Use read_file, grep, and/or list_dir to check blast "
    "radius: tests that inventory workflow or other filenames, README and "
    "code-map docs, and sibling CI files. Then finish with the JSON object "
    '{"findings": [...]}.'
)
BUDGET_EXHAUSTED_NOTICE = (
    "Tool budget exhausted. Finish now with the JSON object "
    '{"findings": [...]} and no further tool calls.'
)
FINALIZE_RETRY_NOTICE = (
    "Your last message was not the required JSON object. Return ONLY the JSON "
    'object {"findings": [...]} now, with no commentary and no tool calls.'
)
# Bound on stub-repair rounds if a model keeps emitting tool calls after the
# tool budget was withdrawn.
MAX_REPAIR_ROUNDS = 3


ChatFn = Callable[[dict[str, Any]], dict[str, Any]]
SleepFn = Callable[[float], None]


def require_openrouter_key(env: dict[str, str]) -> str:
    key = (env.get("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise ActionError(
            "OPENROUTER_API_KEY is missing. Set it on the job env "
            "(not as an action input) from a repository secret."
        )
    return key


def parse_max_tool_turns(
    raw: str | None,
    *,
    default: int = DEFAULT_MAX_TOOL_TURNS,
    what: str = "max_tool_turns",
) -> int:
    """Parse a tool-turn budget. 0 disables tools. Empty uses `default`."""
    text = (raw or "").strip()
    if not text:
        return default
    try:
        value = int(text)
    except ValueError as exc:
        raise ActionError(f"{what} must be an integer") from exc
    if value < 0 or value > MAX_TOOL_TURNS_LIMIT:
        raise ActionError(f"{what} must be 0 through {MAX_TOOL_TURNS_LIMIT}")
    return value


def openrouter_chat(
    api_key: str,
    payload: dict[str, Any],
    *,
    timeout: int,
    sleep: SleepFn = time.sleep,
    stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    attempt = 0
    while True:
        attempt += 1
        request = urllib.request.Request(
            OPENROUTER_URL,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": HTTP_REFERER,
                "X-Title": APP_TITLE,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")[:800]
            if exc.code in RETRYABLE_STATUS and attempt < MAX_HTTP_ATTEMPTS:
                _count_retry(stats)
                sleep(_retry_delay(attempt, exc.headers.get("Retry-After")))
                continue
            raise LaneError(f"OpenRouter HTTP {exc.code}: {redact(err_body)}") from exc
        except TimeoutError as exc:
            if attempt < MAX_HTTP_ATTEMPTS:
                _count_retry(stats)
                sleep(_retry_delay(attempt, None))
                continue
            raise LaneError("OpenRouter request timed out") from exc
        except urllib.error.URLError as exc:
            if attempt < MAX_HTTP_ATTEMPTS:
                _count_retry(stats)
                sleep(_retry_delay(attempt, None))
                continue
            raise LaneError(f"OpenRouter request failed: {redact(str(exc.reason))}") from exc
        except (http.client.HTTPException, OSError) as exc:
            # A connection that drops mid-body raises from response.read() as
            # http.client.IncompleteRead / RemoteDisconnected or a bare
            # ConnectionError — none of which are URLError — and is exactly as
            # transient as a 502. Common on flaky routes (free tiers).
            if attempt < MAX_HTTP_ATTEMPTS:
                _count_retry(stats)
                sleep(_retry_delay(attempt, None))
                continue
            raise LaneError(
                f"OpenRouter connection failed mid-response: {redact(str(exc))}"
            ) from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LaneError(f"OpenRouter returned non-JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LaneError("OpenRouter returned a non-object JSON body")
    return parsed


def _count_retry(stats: dict[str, int] | None) -> None:
    if stats is not None:
        stats["retries"] = stats.get("retries", 0) + 1


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return max(0.0, min(float(retry_after), MAX_RETRY_AFTER_SECONDS))
        except ValueError:
            pass
    return BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))


def run_lane(
    *,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str,
    workspace: Path | None,
    max_tool_turns: int = MAX_TOOL_TURNS,
    effort: str = "",
    timeout: int = DEFAULT_TIMEOUT,
    provider_order: list[str] | None = None,
    # Anchor-gate reference tree for tool-less runs (workspace None): a full
    # checkout of the reviewed head, used only for path/line existence checks.
    anchor_root: Path | None = None,
    chat: ChatFn | None = None,
    expect_coverage: bool = False,
    expect_resolutions: bool = False,
    expected_paths: set[str] | None = None,
    expected_resolution_ids: set[str] | None = None,
) -> LaneResult:
    started = time.monotonic()
    stats: dict[str, int] = {}
    send = chat or (
        lambda payload: openrouter_chat(api_key, payload, timeout=timeout, stats=stats)
    )
    conversation = list(messages)
    tools = list(READ_ONLY_TOOLS) if workspace is not None and max_tool_turns > 0 else None
    usage: dict[str, int | float] = {}
    meta: dict[str, str] = {}

    def _validate(content: str) -> tuple[list[Finding], list[Resolution], list[tuple[str, int]]]:
        findings, resolutions, coverage = parse_lane_payload(
            content,
            model,
            expect_coverage=expect_coverage,
            expect_resolutions=expect_resolutions,
        )
        if expect_coverage and expected_paths:
            problem = validate_coverage(coverage, set(expected_paths))
            if problem:
                raise LaneError(problem)
        if expect_resolutions and expected_resolution_ids:
            provided = {resolution.id for resolution in resolutions}
            missing = sorted(expected_resolution_ids - provided)
            if missing:
                named = ", ".join(missing[:5])
                raise LaneError(
                    f"resolutions are incomplete; missing entries for carried "
                    f"finding(s): {named}"
                )
        return findings, resolutions, coverage

    try:
        content = _run_loop(
            model=model,
            conversation=conversation,
            tools=tools,
            workspace=workspace,
            max_tool_turns=max_tool_turns,
            effort=effort,
            send=send,
            usage=usage,
            stats=stats,
            meta=meta,
            provider_order=provider_order,
            response_schema=findings_json_schema(
                include_coverage=expect_coverage,
                include_resolutions=expect_resolutions,
            ),
            validate_final=_validate,
        )
        findings, resolutions, coverage = _validate(content)
        gate_root = workspace if workspace is not None else anchor_root
        if gate_root is not None:
            findings = sanitize_anchors(findings, gate_root)
    except LaneError as exc:
        failed = failed_lane(model, redact(str(exc)), elapsed_ms=_elapsed_ms(started))
        _attach_stats(failed, stats, usage)
        failed.provider = meta.get("provider")
        return failed

    result = LaneResult(
        schema_version=SCHEMA_VERSION,
        ok=True,
        model=model,
        findings=findings,
        error=None,
        elapsed_ms=_elapsed_ms(started),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        provider=meta.get("provider"),
        resolutions=resolutions,
        coverage=coverage,
    )
    _attach_stats(result, stats, usage)
    return result


def _attach_stats(
    result: LaneResult, stats: dict[str, int], usage: dict[str, int | float]
) -> None:
    result.requests = stats.get("requests")
    result.tool_rounds = stats.get("tool_rounds")
    result.retries = stats.get("retries")
    result.cached_tokens = usage.get("cached_tokens")
    cost = usage.get("cost_usd")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        result.cost_usd = float(cost)
    result.salvaged = bool(stats.get("salvaged"))
    if result.prompt_tokens is None:
        result.prompt_tokens = usage.get("prompt_tokens")
    if result.completion_tokens is None:
        result.completion_tokens = usage.get("completion_tokens")


def _run_loop(
    *,
    model: str,
    conversation: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    workspace: Path | None,
    max_tool_turns: int,
    effort: str,
    send: ChatFn,
    usage: dict[str, int | float],
    stats: dict[str, int] | None = None,
    meta: dict[str, str] | None = None,
    provider_order: list[str] | None = None,
    response_schema: dict[str, Any] | None = None,
    validate_final: Callable[[str], object] | None = None,
) -> str:
    payload_base: dict[str, Any] = {
        "model": model,
        "response_format": {
            "type": "json_schema",
            "json_schema": response_schema or findings_json_schema(),
        },
        # Ask OpenRouter to return the credit cost of each request so the
        # posted review can report what the run actually spent.
        "usage": {"include": True},
    }
    if effort:
        payload_base["reasoning"] = {"effort": effort}
    if provider_order:
        # Pin OpenRouter's provider routing (e.g. for provider bake-offs).
        # No fallbacks: a pinned comparison must not silently reroute.
        payload_base["provider"] = {"order": list(provider_order), "allow_fallbacks": False}

    turns = 0
    repairs = 0
    observation_bytes = 0
    last_error: LaneError | None = None
    tools_active = bool(tools) and workspace is not None
    # JSON schema on the first tool-enabled turn pushes a glance-and-clean
    # empty findings object. Offer tools without a schema until the model
    # stops calling them (or the budget is gone).
    use_schema = not tools_active
    nudged = False
    force_tool = False
    finalize_retried = False
    salvage_attempted = False
    try:
        while True:
            payload = {
                **payload_base,
                "messages": conversation,
            }
            if not use_schema:
                payload.pop("response_format", None)
            if tools_active:
                payload["tools"] = tools
                payload["tool_choice"] = "required" if force_tool else "auto"
            if stats is not None:
                stats["requests"] = stats.get("requests", 0) + 1
            try:
                response = send(payload)
                _absorb_usage(usage, response, meta)
                # In-body errors (HTTP 200 whose JSON carries an error object,
                # a common OpenRouter provider-failure shape) must reach the
                # same schema-fallback and salvage handling as HTTP errors.
                message = _assistant_message(response)
            except LaneError as exc:
                message = str(exc)
                lowered = message.lower()
                schema_rejected = "response_format" in lowered or "json_schema" in lowered
                if use_schema and schema_rejected:
                    use_schema = False
                    last_error = exc
                    continue
                if turns > 0 and not salvage_attempted:
                    # Salvage: a mid-loop failure that survived the HTTP
                    # retries must not discard every gathered observation.
                    # Ask for a final JSON answer from the evidence so far.
                    salvage_attempted = True
                    if stats is not None:
                        stats["salvaged"] = 1
                    if _looks_like_context_overflow(lowered):
                        _shrink_tool_history(conversation)
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "The previous request failed and will not be retried. "
                                "Do not call tools. Return your findings NOW as the "
                                'JSON object {"findings": [...]} from the evidence '
                                "you have already gathered."
                            ),
                        }
                    )
                    tools_active = False
                    use_schema = True
                    continue
                raise
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                force_tool = False
                conversation.append(_assistant_record(message))
                if not tools_active or workspace is None:
                    # The model produced tool calls this loop never solicited
                    # (or cannot service). Answer every call with a stub so
                    # the transcript stays valid, then insist on the finish.
                    repairs += 1
                    if repairs > MAX_REPAIR_ROUNDS:
                        raise LaneError(
                            "model kept issuing tool calls after the tool budget was withdrawn"
                        )
                    for call in tool_calls:
                        conversation.append(_stub_tool_result(call))
                    conversation.append({"role": "user", "content": BUDGET_EXHAUSTED_NOTICE})
                    use_schema = True
                    continue
                turns += 1
                for call in tool_calls:
                    observation = _run_one_tool(workspace, call)
                    conversation.append(observation)
                    text = observation.get("content")
                    if isinstance(text, str):
                        observation_bytes += len(text.encode("utf-8"))
                if turns >= max_tool_turns or observation_bytes >= MAX_OBSERVATION_BYTES:
                    # Withdraw tools BEFORE the next request: the loop must
                    # never solicit a tool call it will not execute (a
                    # dangling assistant tool_calls entry is an invalid
                    # conversation). The observation-byte cap bounds the
                    # resent transcript the same way the turn budget does.
                    tools_active = False
                    use_schema = True
                    conversation.append({"role": "user", "content": BUDGET_EXHAUSTED_NOTICE})
                continue

            content = _message_text(message)
            if tools_active and turns == 0 and not nudged:
                # Glance-and-clean: one forced blast-radius tool pass.
                nudged = True
                force_tool = True
                conversation.append(_assistant_record(message))
                conversation.append({"role": "user", "content": BLAST_RADIUS_NUDGE})
                continue
            if content.strip():
                try:
                    if validate_final is not None:
                        validate_final(content)
                    else:
                        parse_model_findings(content, model)
                except LaneError as exc:
                    if finalize_retried:
                        raise
                    # The tool-backed path runs without response_format, so a
                    # malformed natural finish gets exactly one
                    # schema-enforced, tool-free redo before failing open.
                    finalize_retried = True
                    last_error = exc
                    conversation.append(_assistant_record(message))
                    conversation.append(
                        {
                            "role": "user",
                            "content": f"{FINALIZE_RETRY_NOTICE} Problem: {exc}",
                        }
                    )
                    tools_active = False
                    use_schema = True
                    continue
                return content
            if last_error:
                raise last_error
            raise LaneError("OpenRouter returned an empty assistant message")
    finally:
        if stats is not None:
            stats["tool_rounds"] = turns


def _looks_like_context_overflow(lowered_error: str) -> bool:
    if "prompt is too long" in lowered_error or "too many tokens" in lowered_error:
        return True
    return "context" in lowered_error and (
        "length" in lowered_error
        or "exceed" in lowered_error
        or "window" in lowered_error
        or "token" in lowered_error
    )


def _shrink_tool_history(
    conversation: list[dict[str, Any]], *, keep_last: int = 2, keep_bytes: int = 2_000
) -> None:
    """Truncate old tool observations in place so a salvage request can fit."""
    tool_indexes = [
        index for index, message in enumerate(conversation) if message.get("role") == "tool"
    ]
    shrinkable = tool_indexes[:-keep_last] if keep_last else tool_indexes
    for index in shrinkable:
        content = conversation[index].get("content")
        if not isinstance(content, str):
            continue
        data = content.encode("utf-8")
        if len(data) <= keep_bytes:
            continue
        clipped = data[:keep_bytes].decode("utf-8", errors="ignore")
        conversation[index] = {
            **conversation[index],
            "content": clipped + "\n[observation truncated after a context overflow]",
        }


def _assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    error = response.get("error")
    if isinstance(error, dict):
        raise LaneError(redact(str(error.get("message") or error)))
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LaneError("OpenRouter response has no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise LaneError("OpenRouter choice is not an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise LaneError("OpenRouter choice is missing a message")
    return message


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _assistant_record(message: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        record["tool_calls"] = message["tool_calls"]
    # OpenRouter's reasoning contract: reasoning_details must be passed back
    # unmodified when continuing a tool-calling conversation. Dropping it
    # strips the model's prior reasoning from every later turn. `reasoning`
    # is the normalized text form some providers return instead.
    if message.get("reasoning_details"):
        record["reasoning_details"] = message["reasoning_details"]
    elif message.get("reasoning"):
        record["reasoning"] = message["reasoning"]
    return record


def _stub_tool_result(call: object) -> dict[str, Any]:
    call_id = "unknown"
    if isinstance(call, dict):
        call_id = str(call.get("id") or "unknown")
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": "error: tool budget exhausted; this call was not executed",
    }


def _run_one_tool(workspace: Path, call: object) -> dict[str, Any]:
    if not isinstance(call, dict):
        return {"role": "tool", "tool_call_id": "unknown", "content": "error: malformed tool call"}
    call_id = str(call.get("id") or "unknown")
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = str(function.get("name") or "")
    raw_args = function.get("arguments") or "{}"
    try:
        arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be an object")
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": f"error: invalid tool arguments: {exc}",
        }
    result = dispatch_tool(workspace, name, arguments)
    return {"role": "tool", "tool_call_id": call_id, "content": result}


def _absorb_usage(
    usage: dict[str, int | float],
    response: dict[str, Any],
    meta: dict[str, str] | None = None,
) -> None:
    if meta is not None:
        provider = response.get("provider")
        if isinstance(provider, str) and provider.strip():
            # Last response wins; OpenRouter may reroute between requests.
            meta["provider"] = provider.strip()
    block = response.get("usage")
    if not isinstance(block, dict):
        return
    for key in ("prompt_tokens", "completion_tokens"):
        value = block.get(key)
        if isinstance(value, int):
            usage[key] = usage.get(key, 0) + value
    # OpenRouter credits, plus the provider-billed upstream cost for BYOK
    # keys (where OpenRouter's own `cost` is 0 and the real spend lives in
    # cost_details.upstream_inference_cost).
    cost_details = block.get("cost_details")
    upstream = (
        cost_details.get("upstream_inference_cost")
        if isinstance(cost_details, dict)
        else None
    )
    for cost in (block.get("cost"), upstream):
        if (
            isinstance(cost, (int, float))
            and not isinstance(cost, bool)
            and math.isfinite(cost)
            and cost >= 0
        ):
            usage["cost_usd"] = usage.get("cost_usd", 0) + cost
    details = block.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
        if isinstance(cached, int):
            usage["cached_tokens"] = usage.get("cached_tokens", 0) + cached


def sanitize_anchors(findings: list[Finding], workspace: Path) -> list[Finding]:
    """Deterministic anchor sanity gate: null objectively impossible locations.

    A finding whose path is not tracked at the reviewed commit becomes
    body-only; a line beyond the end of its materialized file loses the line
    anchor. The finding itself always survives. The gate errs toward keeping
    anchors: the inert snapshot deliberately omits oversized and non-regular
    files, so path existence is judged against the commit's tracked-path
    manifest when one is present (falling back to the filesystem, where a
    directory also counts as existing), and the line check runs only on files
    that were actually materialized, counted the way the read-only tools
    number lines. Each adjustment is logged.
    """
    from dataclasses import replace as _replace

    from or_pr_review.workspace import tracked_paths

    manifest = tracked_paths(workspace)
    sanitized: list[Finding] = []
    for finding in findings:
        if finding.file is None:
            sanitized.append(finding)
            continue
        target = workspace / finding.file
        exists = (
            finding.file in manifest if manifest is not None else target.exists()
        ) or target.exists()
        if not exists:
            print(
                f"anchor gate: `{finding.file}` is not tracked at the reviewed "
                f"commit; finding {finding.title[:60]!r} becomes body-only"
            )
            sanitized.append(_replace(finding, file=None, line=None))
            continue
        if finding.line is not None and target.is_file():
            line_count: int | None
            try:
                with target.open("rb") as handle:
                    # Count lines exactly the way read_file/grep number them
                    # (str.splitlines also splits U+2028/NEL/bare \r).
                    line_count = len(
                        handle.read().decode("utf-8", errors="replace").splitlines()
                    )
            except OSError:
                line_count = None
            if line_count is not None and finding.line > line_count:
                print(
                    f"anchor gate: line {finding.line} is beyond the end of "
                    f"`{finding.file}` ({line_count} line(s)); dropping the line "
                    f"anchor of {finding.title[:60]!r}"
                )
                sanitized.append(_replace(finding, line=None))
                continue
        sanitized.append(finding)
    return sanitized


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def response_message_text(response: dict[str, Any]) -> str:
    """Public wrapper for parsing a chat-completions assistant message."""
    return _message_text(_assistant_message(response))
