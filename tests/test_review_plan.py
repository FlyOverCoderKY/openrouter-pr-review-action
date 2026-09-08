from __future__ import annotations

import json

import pytest

from or_pr_review.errors import ActionError
from or_pr_review.review_plan import parse_review_profiles, resolve_review_plan


def _raw(*, deep: bool = True) -> str:
    standard = {
        "lanes": [
            {"model": "openai/gpt-5", "required": True},
            {"model": "anthropic/claude-sonnet-4.6", "required": False},
        ],
        "effort": "low",
        "verify_effort": "medium",
        "max_tool_turns": 50,
        "verify_max_tool_turns": 30,
        "lane_timeout_seconds": 600,
        "job_budget_seconds": 1320,
    }
    profiles: dict[str, object] = {"code": {"standard": standard}}
    if deep:
        profiles["code"] = {
            "standard": standard,
            "deep": {
                **standard,
                "lanes": [
                    {"model": "openai/gpt-5", "required": True},
                    {"model": "anthropic/claude-sonnet-4.6", "required": True},
                    {"model": "google/gemini-3.1-pro", "required": False},
                ],
                "effort": "high",
                "verify_effort": "high",
                "max_tool_turns": 70,
                "verify_max_tool_turns": 40,
                "lane_timeout_seconds": 700,
            },
        }
    return json.dumps({"version": 1, "profiles": profiles})


def _resolve(registry, **changes):
    options = {
        "profile": "code",
        "minimum": "standard",
        "requested_level": "auto",
        "mode": "initial",
        "models": ["x-ai/grok-4.6"],
        "judge_model": "openai/gpt-5.6-luna",
        "routes": {},
        "effort": "",
        "max_tool_turns": 100,
        "job_budget_seconds": 1400,
        "lane_timeout_seconds": 800,
    }
    options.update(changes)
    return resolve_review_plan(registry, **options)


def test_parser_rejects_closed_schema_types_duplicates_and_bounds() -> None:
    invalid = [
        '{"version":1,"profiles":{},"unknown":true}',
        '{"version":1,"version":1,"profiles":{}}',
        '{"version":1,"profiles":{"Code":{"standard":{"lanes":[]}}}}',
        '{"version":1,"profiles":{"code":{"standard":{"lanes":['
        '{"model":" openai/gpt-5","required":true}]}}}}',
        '{"version":1,"profiles":{"code":{"standard":{"lanes":['
        '{"model":"openai/gpt-5","required":1}]}}}}',
        '{"version":1,"profiles":{"code":{"standard":{"lanes":['
        '{"model":"openai/gpt-5","required":true}],"max_tool_turns":true}}}}',
        '{"version":1,"profiles":{"code":{"standard":{"lanes":['
        '{"model":"openai/gpt-5","required":true}],"extra":1}}}}',
    ]
    for raw in invalid:
        with pytest.raises(ActionError):
            parse_review_profiles(raw)
    assert parse_review_profiles(None) is None
    assert parse_review_profiles("  ") is None
    with pytest.raises(ActionError, match="32 KiB"):
        parse_review_profiles(" " + "x" * (33 * 1024))
    with pytest.raises(ActionError, match="32 KiB"):
        parse_review_profiles(" " * (33 * 1024))
    with pytest.raises(ActionError, match="invalid JSON"):
        parse_review_profiles('{"version": NaN, "profiles": {}}')


def test_deep_must_remain_monotonic_and_retain_required_lanes() -> None:
    payload = json.loads(_raw())
    deep = payload["profiles"]["code"]["deep"]
    deep["lanes"][0]["required"] = False
    with pytest.raises(ActionError, match="retain every required"):
        parse_review_profiles(json.dumps(payload))
    payload = json.loads(_raw())
    payload["profiles"]["code"]["deep"]["verify_effort"] = "low"
    with pytest.raises(ActionError, match="must not lower verify_effort"):
        parse_review_profiles(json.dumps(payload))


def test_deep_effort_requires_both_omitted_or_both_explicit() -> None:
    payload = json.loads(_raw())
    payload["profiles"]["code"]["standard"]["effort"] = ""
    with pytest.raises(ActionError, match="effort must both be omitted or both explicit"):
        parse_review_profiles(json.dumps(payload))
    payload = json.loads(_raw())
    payload["profiles"]["code"]["deep"]["verify_effort"] = ""
    with pytest.raises(ActionError, match="verify_effort must both be omitted or both explicit"):
        parse_review_profiles(json.dumps(payload))


def test_deep_accepts_both_omitted_effort() -> None:
    payload = json.loads(_raw())
    payload["profiles"]["code"]["standard"].pop("effort", None)
    payload["profiles"]["code"]["deep"].pop("effort", None)
    registry = parse_review_profiles(json.dumps(payload))
    assert registry is not None


def test_deep_rejects_explicit_effort_reduction() -> None:
    payload = json.loads(_raw())
    payload["profiles"]["code"]["deep"]["effort"] = "minimal"
    with pytest.raises(ActionError, match="must not lower effort"):
        parse_review_profiles(json.dumps(payload))


def test_deep_rejects_lower_job_budget_seconds() -> None:
    payload = json.loads(_raw())
    payload["profiles"]["code"]["deep"]["job_budget_seconds"] = 1319
    with pytest.raises(ActionError, match="must not lower job_budget_seconds"):
        parse_review_profiles(json.dumps(payload))


def test_parser_rejects_oversize_model_slug() -> None:
    oversize = "openai/" + "a" * 200
    payload = {
        "version": 1,
        "profiles": {
            "code": {
                "standard": {
                    "lanes": [{"model": oversize, "required": True}],
                }
            }
        },
    }
    with pytest.raises(ActionError, match="exceeds 200 UTF-8 bytes"):
        parse_review_profiles(json.dumps(payload))


def test_registry_resolves_level_trigger_mode_and_required_lanes() -> None:
    registry = parse_review_profiles(_raw())
    standard = _resolve(registry, mode="verify")
    assert (standard.level, standard.trigger, standard.effort, standard.max_tool_turns) == (
        "standard",
        "baseline",
        "medium",
        30,
    )
    deep = _resolve(registry, requested_level="deep")
    assert deep.level == "deep" and deep.trigger == "manual"
    assert [lane.required for lane in deep.lanes] == [True, True, False]
    assert _resolve(registry, minimum="deep").trigger == "policy"


def test_missing_deep_and_unknown_registry_profile_fail() -> None:
    registry = parse_review_profiles(_raw(deep=False))
    with pytest.raises(ActionError, match="no deep mapping"):
        _resolve(registry, requested_level="deep")
    with pytest.raises(ActionError, match="not configured"):
        _resolve(registry, profile="docs")


def test_legacy_standard_preserves_any_success_lanes_and_rejects_deep() -> None:
    plan = _resolve(None, models=["x-ai/grok-4.6", "openai/gpt-5"])
    assert plan.legacy and not any(lane.required for lane in plan.lanes)
    with pytest.raises(ActionError, match="no deep mapping"):
        _resolve(None, requested_level="deep")
    with pytest.raises(ActionError, match="does not define profile"):
        _resolve(None, profile="security")


def test_routes_are_exactly_bound_and_digest_is_stable_and_sensitive() -> None:
    registry = parse_review_profiles(_raw())
    routes = {
        "openai/gpt-5": {"provider_order": ["OpenAI"], "service_tier": "priority"},
        "unused/model": {"provider_order": ["Other"]},
    }
    one = _resolve(registry, routes=routes)
    two = _resolve(registry, routes=dict(reversed(list(routes.items()))))
    assert one.registry_digest == two.registry_digest
    assert (one.lanes[0].provider, one.lanes[0].service_tier) == ("OpenAI", "priority")
    assert one.lanes[1].provider is None
    changed_route = _resolve(
        registry,
        routes={
            "openai/gpt-5": {"provider_order": ["OpenAI"], "service_tier": "flex"},
            "unused/model": {"provider_order": ["Other"]},
        },
    )
    assert changed_route.registry_digest != one.registry_digest
    payload = json.loads(_raw())
    payload["profiles"]["code"]["deep"]["effort"] = "xhigh"
    changed_registry = _resolve(parse_review_profiles(json.dumps(payload)), routes=routes)
    assert changed_registry.registry_digest != one.registry_digest


def test_selected_panel_must_fit_caller_ceilings() -> None:
    registry = parse_review_profiles(_raw())
    with pytest.raises(ActionError, match="max_tool_turns exceeds"):
        _resolve(registry, max_tool_turns=49)
    with pytest.raises(ActionError, match="job_budget_seconds exceeds"):
        _resolve(registry, job_budget_seconds=1319)
    with pytest.raises(ActionError, match="lane_timeout_seconds exceeds"):
        _resolve(registry, lane_timeout_seconds=599)
