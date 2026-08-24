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
MAX_TOOL_TURNS = 8


ChatFn = Callable[[dict[str, Any]], dict[str, Any]]


def require_openrouter_key(env: dict[str, str]) -> str:
    key = (env.get("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise ActionError(
            "OPENROUTER_API_KEY is missing. Set it on the job env "
            "(not as an action input) from a repository secret."
        )
    return key


def openrouter_chat(api_key: str, payload: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
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
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")[:800]
        raise LaneError(f"OpenRouter HTTP {exc.code}: {redact(err_body)}") from exc
    except urllib.error.URLError as exc:
        raise LaneError(f"OpenRouter request failed: {redact(str(exc.reason))}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LaneError(f"OpenRouter returned non-JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LaneError("OpenRouter returned a non-object JSON body")
    return parsed


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
    send = chat or (lambda payload: openrouter_chat(api_key, payload, timeout=timeout))
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
        )
        findings = parse_model_findings(content, model)
    except LaneError as exc:
        return failed_lane(model, redact(str(exc)), elapsed_ms=_elapsed_ms(started))

    return LaneResult(
        schema_version=SCHEMA_VERSION,
        ok=True,
        model=model,
        findings=findings,
        error=None,
        elapsed_ms=_elapsed_ms(started),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


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
) -> str:
    payload_base: dict[str, Any] = {
        "model": model,
        "response_format": {"type": "json_schema", "json_schema": findings_json_schema()},
    }
    if effort:
        payload_base["reasoning"] = {"effort": effort}

    turns = 0
    last_error: LaneError | None = None
    use_schema = True
    while True:
        payload = {
            **payload_base,
            "messages": conversation,
        }
        if not use_schema:
            payload.pop("response_format", None)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        try:
            response = send(payload)
        except LaneError as exc:
            message = str(exc)
            if use_schema and ("response_format" in message.lower() or "json_schema" in message.lower()):
                use_schema = False
                last_error = exc
                continue
            raise
        _absorb_usage(usage, response)
        message = _assistant_message(response)
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            turns += 1
            if workspace is None or turns > max_tool_turns:
                conversation.append(_assistant_record(message))
                conversation.append(
                    {
                        "role": "user",
                        "content": (
                            "Tool budget exhausted. Finish now with the JSON object "
                            '{"findings": [...]} and no further tool calls.'
                        ),
                    }
                )
                tools = None
                continue
            conversation.append(_assistant_record(message))
            for call in tool_calls:
                conversation.append(_run_one_tool(workspace, call))
            continue

        content = _message_text(message)
        if content.strip():
            return content
        if last_error:
            raise last_error
        raise LaneError("OpenRouter returned an empty assistant message")


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
    return record


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


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def response_message_text(response: dict[str, Any]) -> str:
    """Public wrapper for parsing a chat-completions assistant message."""
    return _message_text(_assistant_message(response))
