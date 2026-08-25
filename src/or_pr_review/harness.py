"""Thin OpenRouter chat-completions harness with optional read-only tools."""

from __future__ import annotations

import json
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
    LaneResult,
    failed_lane,
    findings_json_schema,
    parse_model_findings,
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
    chat: ChatFn | None = None,
) -> LaneResult:
    started = time.monotonic()
    stats: dict[str, int] = {}
    send = chat or (
        lambda payload: openrouter_chat(api_key, payload, timeout=timeout, stats=stats)
    )
    conversation = list(messages)
    tools = list(READ_ONLY_TOOLS) if workspace is not None and max_tool_turns > 0 else None
    usage: dict[str, int] = {}

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
        )
        findings = parse_model_findings(content, model)
    except LaneError as exc:
        failed = failed_lane(model, redact(str(exc)), elapsed_ms=_elapsed_ms(started))
        _attach_stats(failed, stats, usage)
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
    )
    _attach_stats(result, stats, usage)
    return result


def _attach_stats(result: LaneResult, stats: dict[str, int], usage: dict[str, int]) -> None:
    result.requests = stats.get("requests")
    result.tool_rounds = stats.get("tool_rounds")
    result.retries = stats.get("retries")
    result.cached_tokens = usage.get("cached_tokens")
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
    usage: dict[str, int],
    stats: dict[str, int] | None = None,
) -> str:
    payload_base: dict[str, Any] = {
        "model": model,
        "response_format": {"type": "json_schema", "json_schema": findings_json_schema()},
    }
    if effort:
        payload_base["reasoning"] = {"effort": effort}

    turns = 0
    repairs = 0
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
            _absorb_usage(usage, response)
            message = _assistant_message(response)
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
                    conversation.append(_run_one_tool(workspace, call))
                if turns >= max_tool_turns:
                    # Withdraw tools BEFORE the next request: the loop must
                    # never solicit a tool call it will not execute (a
                    # dangling assistant tool_calls entry is an invalid
                    # conversation).
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
                    conversation.append({"role": "user", "content": FINALIZE_RETRY_NOTICE})
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


def _absorb_usage(usage: dict[str, int], response: dict[str, Any]) -> None:
    block = response.get("usage")
    if not isinstance(block, dict):
        return
    for key in ("prompt_tokens", "completion_tokens"):
        value = block.get(key)
        if isinstance(value, int):
            usage[key] = usage.get(key, 0) + value
    details = block.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
        if isinstance(cached, int):
            usage["cached_tokens"] = usage.get("cached_tokens", 0) + cached


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def response_message_text(response: dict[str, Any]) -> str:
    """Public wrapper for parsing a chat-completions assistant message."""
    return _message_text(_assistant_message(response))
