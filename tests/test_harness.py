from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from or_pr_review.errors import ActionError
from or_pr_review.harness import (
    BLAST_RADIUS_NUDGE,
    DEFAULT_MAX_TOOL_TURNS,
    MAX_TOOL_TURNS,
    parse_max_tool_turns,
    require_openrouter_key,
    run_lane,
)


def test_require_key_fail_closed() -> None:
    with pytest.raises(ActionError, match="OPENROUTER_API_KEY"):
        require_openrouter_key({})


def test_lane_parses_structured_findings(tmp_path: Path) -> None:
    def chat(_payload: dict) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"findings":[{"title":"Race","body":"check-then-act",'
                            '"severity":"risk","file":"db.py","line":9}]}'
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        max_tool_turns=0,
    )
    assert result.ok
    assert result.findings[0].title == "Race"
    assert result.prompt_tokens == 10


def test_lane_tool_call_then_findings(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    calls = {"n": 0}

    def chat(payload: dict) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"a.py"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"findings":[]}',
                    }
                }
            ]
        }

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        max_tool_turns=4,
    )
    assert result.ok
    assert result.findings == []
    assert calls["n"] == 2


def test_lane_bad_json_fail_opens() -> None:
    def chat(_payload: dict) -> dict:
        return {"choices": [{"message": {"content": "sorry I cannot"}}]}

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=None,
        chat=chat,
    )
    assert not result.ok
    assert result.findings == []
    assert result.error


def test_lane_http_style_error_fail_opens() -> None:
    from or_pr_review.errors import LaneError

    def chat(_payload: dict) -> dict:
        raise LaneError("OpenRouter HTTP 429: rate limited")

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=None,
        chat=chat,
    )
    assert not result.ok
    assert "429" in (result.error or "")


def test_default_first_pass_tool_budget_is_fifty() -> None:
    assert DEFAULT_MAX_TOOL_TURNS == 50
    assert MAX_TOOL_TURNS == 50
    assert inspect.signature(run_lane).parameters["max_tool_turns"].default == 50
    assert parse_max_tool_turns(None) == 50
    assert parse_max_tool_turns("") == 50
    assert parse_max_tool_turns("30") == 30
    assert parse_max_tool_turns("0") == 0
    with pytest.raises(ActionError, match="integer"):
        parse_max_tool_turns("nope")
    with pytest.raises(ActionError, match="0 through"):
        parse_max_tool_turns("-1")
    with pytest.raises(ActionError, match="0 through"):
        parse_max_tool_turns("1001")


def _tool_reply() -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"a.py"}',
                            },
                        }
                    ],
                }
            }
        ]
    }


def _findings_reply() -> dict:
    return {"choices": [{"message": {"content": '{"findings":[]}'}}]}


def test_tools_enabled_omits_json_schema_on_first_turn(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return _tool_reply()
        return _findings_reply()

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        max_tool_turns=50,
    )
    assert result.ok
    assert "response_format" not in payloads[0]
    assert payloads[0].get("tools")
    assert payloads[0].get("tool_choice") == "auto"


def test_zero_tool_finish_is_nudged_then_required(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return _findings_reply()
        if len(payloads) == 2:
            assert any(
                item.get("content") == BLAST_RADIUS_NUDGE for item in payload["messages"]
            )
            assert payload.get("tool_choice") == "required"
            assert "response_format" not in payload
            return _tool_reply()
        return _findings_reply()

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
    )
    assert result.ok
    assert len(payloads) == 3


def test_default_budget_allows_more_than_legacy_eight_turns(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    calls = {"n": 0}

    def chat(_payload: dict) -> dict:
        calls["n"] += 1
        if calls["n"] <= 12:
            return _tool_reply()
        return _findings_reply()

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
    )
    assert result.ok
    assert calls["n"] == 13


def test_schema_used_when_tools_disabled() -> None:
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        return _findings_reply()

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=None,
        chat=chat,
        max_tool_turns=50,
    )
    assert result.ok
    assert "response_format" in payloads[0]
    assert "tools" not in payloads[0]
