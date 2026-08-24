from __future__ import annotations

from pathlib import Path

import pytest

from or_pr_review.errors import ActionError
from or_pr_review.harness import require_openrouter_key, run_lane


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
