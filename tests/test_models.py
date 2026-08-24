from __future__ import annotations

import json

import pytest

from or_pr_review.errors import ActionError
from or_pr_review.models import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MODEL,
    LANE_CAP,
    judge_is_needed,
    matrix_json,
    parse_judge_model,
    parse_models,
)


def test_empty_models_uses_default_grok_slug() -> None:
    assert parse_models("") == [DEFAULT_MODEL]
    assert parse_models(None) == ["x-ai/grok-4.6"]


def test_comma_separated_list_trims_and_keeps_order() -> None:
    slugs = parse_models(" x-ai/grok-4.6, anthropic/claude-sonnet-4.6 ,google/gemini-2.5-pro ")
    assert slugs == [
        "x-ai/grok-4.6",
        "anthropic/claude-sonnet-4.6",
        "google/gemini-2.5-pro",
    ]


def test_duplicate_slugs_are_separate_lanes() -> None:
    slugs = parse_models("x-ai/grok-4.6,x-ai/grok-4.6")
    assert slugs == ["x-ai/grok-4.6", "x-ai/grok-4.6"]


def test_variant_suffix_is_allowed() -> None:
    assert parse_models("x-ai/grok-4.6:beta") == ["x-ai/grok-4.6:beta"]


def test_four_lanes_is_at_cap() -> None:
    slugs = parse_models(
        "x-ai/grok-4.6,anthropic/claude-sonnet-4.6,google/gemini-2.5-pro,openai/gpt-5"
    )
    assert len(slugs) == LANE_CAP == 4


def test_five_lanes_fails_clearly() -> None:
    with pytest.raises(ActionError, match="hard cap is 4"):
        parse_models(
            "x-ai/grok-4.6,anthropic/claude-sonnet-4.6,"
            "google/gemini-2.5-pro,openai/gpt-5,meta-llama/llama-4-maverick"
        )


def test_only_commas_is_empty_error() -> None:
    with pytest.raises(ActionError, match="empty after parsing"):
        parse_models(",,,")


def test_invalid_slug_rejected() -> None:
    with pytest.raises(ActionError, match="invalid OpenRouter model slug"):
        parse_models("grok-4.6")
    with pytest.raises(ActionError, match="invalid OpenRouter model slug"):
        parse_models("x-ai/grok 4.6")


def test_judge_model_defaults_and_parses() -> None:
    assert parse_judge_model("") == DEFAULT_JUDGE_MODEL == "google/gemini-3.1-flash-lite"
    assert parse_judge_model("anthropic/claude-haiku-4.5") == "anthropic/claude-haiku-4.5"
    with pytest.raises(ActionError, match="judge_model"):
        parse_judge_model("not-a-slug")


def test_one_lane_skips_judge() -> None:
    assert not judge_is_needed(["x-ai/grok-4.6"])
    assert judge_is_needed(["x-ai/grok-4.6", "openai/gpt-4.1-nano"])


def test_matrix_json_includes_indexes() -> None:
    payload = json.loads(matrix_json(["x-ai/grok-4.6", "anthropic/claude-sonnet-4.6"]))
    assert payload == [
        {"index": 0, "model": "x-ai/grok-4.6"},
        {"index": 1, "model": "anthropic/claude-sonnet-4.6"},
    ]
