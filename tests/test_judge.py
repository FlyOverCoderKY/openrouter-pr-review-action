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
    assert issues[0].title == "Missing auth check"
    assert issues[0].models == ["x-ai/grok-4.6", "anthropic/claude-sonnet-4.6"]


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
                            '"models":["x-ai/grok-4.6"],"sources":["0.0"]}]}'
                        )
                    }
                }
            ],
            # Non-BYOK shape: cost_details mirrors the same charge — it
            # must NOT be double-counted.
            "usage": {
                "cost": 0.0021,
                "is_byok": False,
                "cost_details": {"upstream_inference_cost": 0.0021},
            },
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
    issues, mode, cost = issues
    assert mode == "merged"
    assert cost == pytest.approx(0.0021)
    assert issues[0].title == "Race"
    assert seen["model"] == "google/gemini-3.1-flash-lite"
    assert seen["reasoning"] == {"effort": "minimal"}
    assert seen["usage"] == {"include": True}
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


def test_coverage_repairs_unaccounted_findings() -> None:
    from or_pr_review.judge import run_llm_judge

    lanes = [
        {"model": "x-ai/grok-4.6", "findings": [
            {"title": "Bug A", "body": "a", "severity": "bug", "file": "a.py", "line": 1},
            {"title": "Risk B", "body": "b", "severity": "risk", "file": "b.py", "line": 2},
        ]},
        {"model": "z-ai/glm-5.3-flash", "findings": [
            {"title": "Risk D", "body": "d", "severity": "risk", "file": "d.py", "line": 4},
        ]},
    ]

    def dropping_chat(_payload: dict) -> dict:
        # Judge accounts for 0.0 and 0.1 but silently drops the second
        # lane's 1.0 — the exact under-merge the count floor missed.
        return {
            "choices": [{"message": {"content": (
                '{"issues": ['
                '{"title": "Bug A", "body": "a", "severity": "bug", "file": "a.py", "line": 1,'
                ' "models": ["x-ai/grok-4.6"], "sources": ["0.0"]},'
                '{"title": "Risk B", "body": "b", "severity": "risk", "file": "b.py", "line": 2,'
                ' "models": ["x-ai/grok-4.6"], "sources": ["0.1"]}'
                "]}"
            )}}]
        }

    issues, mode, _cost = run_llm_judge(
        model="google/gemini-3.1-flash-lite", lanes=lanes, api_key="sk-test", chat=dropping_chat
    )
    assert mode == "repaired(+1)"
    assert {i.title for i in issues} == {"Bug A", "Risk B", "Risk D"}
    restored = next(i for i in issues if i.title == "Risk D")
    assert restored.body == "d" and restored.models == ["z-ai/glm-5.3-flash"]


def test_coverage_falls_back_on_untrusted_sources() -> None:
    from or_pr_review.judge import run_llm_judge

    lanes = [
        {"model": "x-ai/grok-4.6", "findings": [
            {"title": "Bug A", "body": "short", "severity": "nit", "file": "a.py", "line": 1},
        ]},
        {"model": "z-ai/glm-5.3-flash", "findings": [
            {"title": "Bug A", "body": "a much longer explanation", "severity": "bug", "file": "a.py", "line": 1},
            {"title": "Risk D", "body": "d", "severity": "risk", "file": "d.py", "line": 4},
        ]},
    ]

    def fabricating_chat(_payload: dict) -> dict:
        # Unknown source id -> the accounting cannot be trusted.
        return {
            "choices": [{"message": {"content": (
                '{"issues": [{"title": "Bug A", "body": "a", "severity": "bug",'
                ' "file": "a.py", "line": 1, "models": ["x-ai/grok-4.6"], "sources": ["9.9"]}]}'
            )}}]
        }

    issues, mode, _cost = run_llm_judge(
        model="google/gemini-3.1-flash-lite", lanes=lanes, api_key="sk-test", chat=fabricating_chat
    )
    assert mode == "union-fallback"
    # Union merges the exact duplicate: strongest severity, longer body,
    # both lanes attributed (the first-wins bug is fixed).
    assert {i.title for i in issues} == {"Bug A", "Risk D"}
    bug_a = next(i for i in issues if i.title == "Bug A")
    assert bug_a.severity == "bug"
    assert bug_a.body == "a much longer explanation"
    assert set(bug_a.models) == {"x-ai/grok-4.6", "z-ai/glm-5.3-flash"}
    # Severity-sorted output: bug before risk.
    assert issues[0].title == "Bug A"


def test_coverage_accepts_a_fully_accounted_merge() -> None:
    from or_pr_review.judge import run_llm_judge

    lanes = [
        {"model": "x-ai/grok-4.6", "findings": [
            {"title": "Bug A", "body": "a", "severity": "bug", "file": "a.py", "line": 1},
        ]},
        {"model": "z-ai/glm-5.3-flash", "findings": [
            {"title": "Bug A variant", "body": "same defect", "severity": "bug", "file": "a.py", "line": 1},
        ]},
    ]

    def merging_chat(_payload: dict) -> dict:
        return {
            "choices": [{"message": {"content": (
                '{"issues": [{"title": "Bug A", "body": "a", "severity": "bug", "file": "a.py",'
                ' "line": 1, "models": ["x-ai/grok-4.6", "z-ai/glm-5.3-flash"],'
                ' "sources": ["0.0", "1.0"]}]}'
            )}}]
        }

    issues, mode, _cost = run_llm_judge(
        model="google/gemini-3.1-flash-lite", lanes=lanes, api_key="sk-test", chat=merging_chat
    )
    assert mode == "merged"
    assert len(issues) == 1


def test_judge_contract_is_identity_tracked_union_merge() -> None:
    from or_pr_review.judge import build_judge_messages

    lanes = [{"model": "m", "findings": [{"title": "T", "body": "b", "severity": "nit", "file": None, "line": None}]}]
    messages = build_judge_messages(lanes)
    system, user = messages[0]["content"], messages[1]["content"]
    assert "UNION-MERGE" in system
    assert "ACCOUNT FOR EVERY input id" in system
    assert "keep them as separate issues" in system
    assert "Never drop" in system
    # The user message must not carry the old filter-style order.
    assert "de-dupe" not in user.lower()
    assert "Union-merge" in user
    assert "exactly one output issue" in user
    # Input findings are id-annotated.
    assert '"id": "0.0"' in user


def test_over_broad_merge_is_split_back() -> None:
    from or_pr_review.judge import run_llm_judge

    lanes = [
        {"model": "x-ai/grok-4.6", "findings": [
            {"title": "Bug A", "body": "a", "severity": "bug", "file": "a.py", "line": 1},
            {"title": "Risk D", "body": "d", "severity": "risk", "file": "d.py", "line": 4},
        ]},
    ]

    def lumping_chat(_payload: dict) -> dict:
        # Accounting is "legal" (both ids claimed) but the merge spans two
        # files — distinct findings compressed into one issue.
        return {
            "choices": [{"message": {"content": (
                '{"issues": [{"title": "Various issues", "body": "combined",'
                ' "severity": "bug", "file": "a.py", "line": 1,'
                ' "models": ["x-ai/grok-4.6"], "sources": ["0.0", "0.1"]}]}'
            )}}]
        }

    issues, mode, _cost = run_llm_judge(
        model="google/gemini-3.1-flash-lite", lanes=lanes, api_key="sk-test", chat=lumping_chat
    )
    assert mode == "repaired(split+2)"
    # Both constituent findings restored verbatim; the lump is gone.
    assert {i.title for i in issues} == {"Bug A", "Risk D"}
    assert all(i.body in {"a", "d"} for i in issues)
