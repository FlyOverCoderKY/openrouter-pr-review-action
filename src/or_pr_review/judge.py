"""OpenRouter judge: merge/de-dupe already-structured lane findings.

The judge is not a second reviewer. It is skipped when only one review lane
is configured. Schema mismatches fail the job (fail-closed).
"""

from __future__ import annotations

import json
from typing import Any

from or_pr_review.errors import ActionError, SchemaError
from or_pr_review.harness import ChatFn, openrouter_chat, response_message_text
from or_pr_review.merge import MergedIssue
from or_pr_review.redaction import redact
from or_pr_review.schema import (
    MAX_BODY,
    MAX_FINDINGS,
    MAX_TITLE,
    SEVERITIES,
    extract_json_object,
    valid_review_path,
)

# Cheapest/fastest thinking level Gemini 3.1 Flash Lite documents (minimal/off).
JUDGE_REASONING = {"effort": "minimal"}


def judge_json_schema() -> dict[str, Any]:
    return {
        "name": "merged_pr_review_issues",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "body": {"type": "string"},
                            "severity": {"type": "string", "enum": list(SEVERITIES)},
                            "file": {"type": ["string", "null"]},
                            "line": {"type": ["integer", "null"]},
                            "models": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["title", "body", "severity", "file", "line", "models"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["issues"],
            "additionalProperties": False,
        },
    }


def build_judge_messages(lanes: list[dict[str, Any]]) -> list[dict[str, str]]:
    payload = json.dumps({"lanes": lanes}, indent=2)
    return [
        {
            "role": "system",
            "content": (
                "You merge already-structured pull-request review findings from "
                "independent model lanes. You are not a second reviewer and must "
                "not invent new issues. De-dupe the same underlying problem, keep "
                "the strongest severity, prefer the clearest body, and list every "
                "lane model that reported it in `models`.\n\n"
                "Return JSON only: {\"issues\": [{\"title\", \"body\", \"severity\", "
                "\"file\", \"line\", \"models\"}]}. severity is bug, risk, or nit. "
                "file/line may be null. Do not think at length; this is clerical merge."
            ),
        },
        {
            "role": "user",
            "content": (
                "Merge and de-dupe these lane results. Attribution models must be "
                "slugs from the input lanes.\n\n"
                f"{payload}"
            ),
        },
    ]


def parse_judge_issues(payload: object, *, allowed_models: list[str]) -> list[MergedIssue]:
    """Fail-closed: any schema mismatch raises SchemaError."""
    if isinstance(payload, str):
        try:
            payload = extract_json_object(payload)
        except Exception as exc:  # noqa: BLE001 — treat as schema failure
            raise SchemaError(f"judge output is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SchemaError("judge output must be a JSON object")
    raw_issues = payload.get("issues")
    if raw_issues is None:
        raise SchemaError("judge output is missing an issues array")
    if not isinstance(raw_issues, list):
        raise SchemaError("judge issues must be an array")
    if len(raw_issues) > MAX_FINDINGS:
        raise SchemaError(f"judge returned more than {MAX_FINDINGS} issues")
    allowed = set(allowed_models)
    issues: list[MergedIssue] = []
    for item in raw_issues:
        issues.append(_parse_one_issue(item, allowed=allowed))
    return issues


def _parse_one_issue(raw: object, *, allowed: set[str]) -> MergedIssue:
    if not isinstance(raw, dict):
        raise SchemaError("each judge issue must be an object")
    title = raw.get("title")
    body = raw.get("body")
    severity = raw.get("severity")
    models = raw.get("models")
    if not isinstance(title, str) or not title.strip():
        raise SchemaError("judge issue title is required")
    if not isinstance(body, str) or not body.strip():
        raise SchemaError("judge issue body is required")
    if not isinstance(severity, str) or severity.strip().lower() not in SEVERITIES:
        raise SchemaError(f"judge issue severity must be one of {', '.join(SEVERITIES)}")
    if not isinstance(models, list) or not models or not all(isinstance(m, str) and m.strip() for m in models):
        raise SchemaError("judge issue models must be a non-empty array of slugs")
    cleaned_models: list[str] = []
    for model in models:
        slug = model.strip()
        if allowed and slug not in allowed:
            raise SchemaError(f"judge attributed an issue to unknown lane model {slug!r}")
        if slug not in cleaned_models:
            cleaned_models.append(slug)
    file_value = raw.get("file")
    if file_value is not None and not isinstance(file_value, str):
        raise SchemaError("judge issue file must be a string or null")
    path = file_value.strip() if isinstance(file_value, str) and file_value.strip() else None
    if path and not valid_review_path(path):
        path = None
    line = raw.get("line")
    if line is None or line == "":
        line_n = None
    elif isinstance(line, bool) or not isinstance(line, int) or line <= 0:
        raise SchemaError("judge issue line must be a positive integer or null")
    else:
        line_n = line
    return MergedIssue(
        title=title.strip()[:MAX_TITLE],
        body=body.strip()[:MAX_BODY],
        severity=severity.strip().lower(),
        file=path,
        line=line_n,
        models=cleaned_models,
    )


def run_llm_judge(
    *,
    model: str,
    lanes: list[dict[str, Any]],
    api_key: str,
    timeout: int = 180,
    chat: ChatFn | None = None,
) -> list[MergedIssue]:
    allowed = [str(lane.get("model")) for lane in lanes if isinstance(lane.get("model"), str)]
    send = chat or (lambda payload: openrouter_chat(api_key, payload, timeout=timeout))
    payload: dict[str, Any] = {
        "model": model,
        "messages": build_judge_messages(lanes),
        "response_format": {"type": "json_schema", "json_schema": judge_json_schema()},
        "reasoning": dict(JUDGE_REASONING),
    }
    try:
        response = send(payload)
    except Exception as exc:  # noqa: BLE001
        raise ActionError(f"judge OpenRouter call failed: {redact(str(exc))}") from exc
    try:
        content = response_message_text(response)
    except Exception as exc:  # noqa: BLE001
        raise SchemaError(f"judge response shape is invalid: {exc}") from exc
    if not content.strip():
        raise SchemaError("judge returned an empty assistant message")
    return parse_judge_issues(content, allowed_models=allowed)
