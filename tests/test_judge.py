from __future__ import annotations

import pytest

from or_pr_review.errors import SchemaError
from or_pr_review.judge import JUDGE_REASONING, parse_judge_issues, run_llm_judge
from or_pr_review.models import (
    DEFAULT_JUDGE_MODEL,
    judge_is_needed,
    parse_judge_model,
)


def test_default_judge_model_is_gemini_flash_lite() -> None:
    assert parse_judge_model("") == "google/gemini-3.1-flash-lite"
    assert DEFAULT_JUDGE_MODEL == "google/gemini-3.1-flash-lite"


def test_judge_needed_only_for_two_plus_lanes() -> None:
    assert not judge_is_needed(["x-ai/grok-4.6"])
    assert judge_is_needed(["x-ai/grok-4.6", "anthropic/claude-sonnet-4.6"])


def test_judge_reasoning_is_minimal() -> None:
    assert JUDGE_REASONING == {"effort": "minimal"}


def test_parse_judge_issues_happy_path() -> None:
    issues = parse_judge_issues(
        {
            "issues": [
                {
                    "title": "Missing auth check",
                    "body": "Unauthenticated POST",
                    "severity": "bug",
                    "file": "src/api.py",
                    "line": 42,
                    "models": ["x-ai/grok-4.6", "anthropic/claude-sonnet-4.6"],
                }
            ]
        },
        allowed_models=["x-ai/grok-4.6", "anthropic/claude-sonnet-4.6"],
    )
    assert len(issues) == 1
    assert issues[0].heading(1) == (
        "Issue 1 - Missing auth check "
        "(identified by x-ai/grok-4.6 and anthropic/claude-sonnet-4.6)"
    )


def test_parse_judge_issues_schema_mismatch_fail_closed() -> None:
    with pytest.raises(SchemaError, match="missing an issues array"):
        parse_judge_issues({"findings": []}, allowed_models=["x-ai/grok-4.6"])
    with pytest.raises(SchemaError, match="unknown lane model"):
        parse_judge_issues(
            {
                "issues": [
                    {
                        "title": "x",
                        "body": "y",
                        "severity": "bug",
                        "file": None,
                        "line": None,
                        "models": ["invented/model"],
                    }
                ]
            },
            allowed_models=["x-ai/grok-4.6"],
        )


def test_run_llm_judge_sends_schema_and_minimal_reasoning() -> None:
    seen: dict = {}

    def chat(payload: dict) -> dict:
        seen.update(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"issues":[{"title":"Race","body":"check-then-act",'
                            '"severity":"risk","file":"db.py","line":9,'
                            '"models":["x-ai/grok-4.6"]}]}'
                        )
                    }
                }
            ]
        }

    issues = run_llm_judge(
        model="google/gemini-3.1-flash-lite",
        lanes=[
            {
                "model": "x-ai/grok-4.6",
                "ok": True,
                # The recall floor/ceiling compares judge output to lane
                # findings, so the merged issue must exist in a lane — a
                # judge cannot invent issues from empty lanes.
                "findings": [
                    {"title": "Race", "body": "check-then-act", "severity": "risk", "file": "db.py", "line": 9}
                ],
                "error": None,
            }
        ],
        api_key="sk-test",
        chat=chat,
    )
    assert issues[0].title == "Race"
    assert seen["model"] == "google/gemini-3.1-flash-lite"
    assert seen["reasoning"] == {"effort": "minimal"}
    assert seen["response_format"]["type"] == "json_schema"


def test_run_llm_judge_bad_output_fail_closed() -> None:
    def chat(_payload: dict) -> dict:
        return {"choices": [{"message": {"content": "not json"}}]}

    with pytest.raises(SchemaError):
        run_llm_judge(
            model="google/gemini-3.1-flash-lite",
            lanes=[{"model": "x-ai/grok-4.6"}],
            api_key="sk-test",
            chat=chat,
        )


def test_recall_floor_falls_back_to_deterministic_union() -> None:
    from or_pr_review.judge import run_llm_judge

    lanes = [
        {
            "model": "x-ai/grok-4.6",
            "findings": [
                {"title": "Bug A", "body": "a", "severity": "bug", "file": "a.py", "line": 1},
                {"title": "Risk B", "body": "b", "severity": "risk", "file": "b.py", "line": 2},
                {"title": "Nit C", "body": "c", "severity": "nit", "file": None, "line": None},
            ],
        },
        {
            "model": "z-ai/glm-5.3-flash",
            "findings": [
                {"title": "Bug A", "body": "a again", "severity": "bug", "file": "a.py", "line": 1},
                {"title": "Risk D", "body": "d", "severity": "risk", "file": "d.py", "line": 4},
            ],
        },
    ]

    def filtering_chat(_payload: dict) -> dict:
        # A judge that filtered down to one issue: below the floor (3).
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"issues": [{"title": "Bug A", "body": "a", "severity": "bug",'
                            ' "file": "a.py", "line": 1, "models": ["x-ai/grok-4.6"]}]}'
                        )
                    }
                }
            ]
        }

    issues = run_llm_judge(
        model="google/gemini-3.1-flash-lite",
        lanes=lanes,
        api_key="sk-test",
        chat=filtering_chat,
    )
    # Fallback union: 4 distinct issues; the exact duplicate Bug A merged
    # with both lanes attributed.
    assert len(issues) == 4
    bug_a = next(i for i in issues if i.title == "Bug A")
    assert set(bug_a.models) == {"x-ai/grok-4.6", "z-ai/glm-5.3-flash"}
    assert {i.title for i in issues} == {"Bug A", "Risk B", "Nit C", "Risk D"}


def test_recall_floor_accepts_a_correct_merge() -> None:
    from or_pr_review.judge import run_llm_judge

    lanes = [
        {"model": "x-ai/grok-4.6", "findings": [
            {"title": "Bug A", "body": "a", "severity": "bug", "file": "a.py", "line": 1},
            {"title": "Risk B", "body": "b", "severity": "risk", "file": "b.py", "line": 2},
        ]},
        {"model": "z-ai/glm-5.3-flash", "findings": [
            {"title": "Bug A variant", "body": "same defect", "severity": "bug", "file": "a.py", "line": 1},
        ]},
    ]

    def merging_chat(_payload: dict) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"issues": ['
                            '{"title": "Bug A", "body": "a", "severity": "bug", "file": "a.py", "line": 1,'
                            ' "models": ["x-ai/grok-4.6", "z-ai/glm-5.3-flash"]},'
                            '{"title": "Risk B", "body": "b", "severity": "risk", "file": "b.py", "line": 2,'
                            ' "models": ["x-ai/grok-4.6"]}'
                            "]}"
                        )
                    }
                }
            ]
        }

    issues = run_llm_judge(
        model="google/gemini-3.1-flash-lite",
        lanes=lanes,
        api_key="sk-test",
        chat=merging_chat,
    )
    assert len(issues) == 2  # floor is 2 (largest lane), merge accepted


def test_judge_contract_is_union_merge() -> None:
    from or_pr_review.judge import build_judge_messages

    system = build_judge_messages([])[0]["content"]
    assert "UNION-MERGE" in system
    assert "MUST contain every distinct finding" in system
    assert "when unsure" in system and "keep both" in system
    assert "Never drop" in system
    assert "at least as" in system
