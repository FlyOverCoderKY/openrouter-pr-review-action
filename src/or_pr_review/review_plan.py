"""Strict trusted review-profile parsing and side-effect-free plan resolution.

The profiles input is workflow configuration, not repository policy.  It is
therefore deliberately small, closed-schema, and independent of GitHub or
provider calls.  ``REVIEW.md`` may select a named profile and a minimum level,
but cannot supply models, routes, or any other execution configuration.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from or_pr_review.errors import ActionError
from or_pr_review.models import DEFAULT_JUDGE_MODEL, parse_slug

_PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_PROVIDER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$")
_MAX_INPUT_BYTES = 32 * 1024
_MAX_DEPTH = 16
_MAX_PROFILES = 8
_MAX_LANES = 4
_EFFORTS = ("", "none", "minimal", "low", "medium", "high", "xhigh")
_EXPLICIT_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
_EXPLICIT_EFFORT_RANK = {value: index for index, value in enumerate(_EXPLICIT_EFFORTS)}
_LEVELS = {"standard", "deep"}
_REQUESTED_LEVELS = {"auto", "deep"}
_MODES = {"initial", "verify"}


@dataclass(frozen=True)
class ReviewLane:
    model: str
    required: bool
    provider: str | None = None
    service_tier: str | None = None


@dataclass(frozen=True)
class ReviewPlan:
    profile: str
    level: str
    trigger: str
    lanes: tuple[ReviewLane, ...]
    judge_model: str
    effort: str
    max_tool_turns: int
    lane_timeout_seconds: int
    job_budget_seconds: int
    registry_digest: str
    legacy: bool


@dataclass(frozen=True)
class _Panel:
    lanes: tuple[ReviewLane, ...]
    judge_model: str
    effort: str
    verify_effort: str
    max_tool_turns: int
    verify_max_tool_turns: int
    lane_timeout_seconds: int
    job_budget_seconds: int


@dataclass(frozen=True)
class _Profile:
    name: str
    standard: _Panel
    deep: _Panel | None


@dataclass(frozen=True)
class ReviewProfileRegistry:
    """Validated, immutable named profile configuration."""

    profiles: tuple[_Profile, ...]

    def profile(self, name: str) -> _Profile | None:
        return next((item for item in self.profiles if item.name == name), None)


def _error(reason: str) -> ActionError:
    return ActionError(f"review_profiles: {reason}")


def _exact_string(value: Any, *, what: str) -> str:
    if type(value) is not str:
        raise _error(f"{what} must be a string")
    return value


def _exact_slug(value: Any, *, what: str) -> str:
    text = _exact_string(value, what=what)
    if text != text.strip():
        raise _error(f"{what} must not contain surrounding whitespace")
    if len(text.encode("utf-8", "strict")) > 200:
        raise _error(f"{what} exceeds 200 UTF-8 bytes")
    try:
        return parse_slug(text, what=what)
    except ActionError as exc:
        raise _error(str(exc)) from exc


def _integer(value: Any, *, what: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise _error(f"{what} must be an integer from {lower} through {upper}")
    return value


def _effort(value: Any, *, what: str) -> str:
    text = _exact_string(value, what=what)
    if text not in _EFFORTS:
        raise _error(f"{what} must be one of empty, none, minimal, low, medium, high, xhigh")
    return text


def _reject_surrogates(value: Any) -> None:
    if isinstance(value, str):
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise _error("contains an invalid Unicode surrogate")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_surrogates(key)
            _reject_surrogates(item)
    elif isinstance(value, list):
        for item in value:
            _reject_surrogates(item)


def _check_depth(value: Any, depth: int = 0) -> None:
    if depth > _MAX_DEPTH:
        raise _error(f"JSON nesting depth exceeds {_MAX_DEPTH}")
    if isinstance(value, dict):
        for item in value.values():
            _check_depth(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _check_depth(item, depth + 1)


def _parse_json(raw: str) -> dict[str, Any]:
    try:
        encoded = raw.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise _error("is not valid UTF-8") from exc
    if len(encoded) > _MAX_INPUT_BYTES:
        raise _error("exceeds 32 KiB UTF-8 bytes")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _error(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=no_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(constant)),
        )
    except (json.JSONDecodeError, RecursionError, UnicodeError, ValueError) as exc:
        if isinstance(exc, ActionError):
            raise
        raise _error(f"invalid JSON: {exc}") from exc
    if type(value) is not dict:
        raise _error("must be a JSON object")
    _reject_surrogates(value)
    _check_depth(value)
    return value


def _parse_lane(value: Any, *, name: str, level: str, index: int) -> ReviewLane:
    where = f"profiles.{name}.{level}.lanes[{index}]"
    if type(value) is not dict or set(value) != {"model", "required"}:
        raise _error(f"{where} must contain exactly model and required")
    required = value["required"]
    if type(required) is not bool:
        raise _error(f"{where}.required must be a boolean")
    return ReviewLane(_exact_slug(value["model"], what=f"{where}.model"), required)


def _parse_panel(value: Any, *, name: str, level: str) -> _Panel:
    where = f"profiles.{name}.{level}"
    allowed = {
        "lanes",
        "judge_model",
        "effort",
        "verify_effort",
        "max_tool_turns",
        "verify_max_tool_turns",
        "lane_timeout_seconds",
        "job_budget_seconds",
    }
    if type(value) is not dict or "lanes" not in value or set(value) - allowed:
        raise _error(f"{where} accepts only lanes and optional panel settings")
    raw_lanes = value["lanes"]
    if type(raw_lanes) is not list or not raw_lanes or len(raw_lanes) > _MAX_LANES:
        raise _error(f"{where}.lanes must contain 1 through {_MAX_LANES} lanes")
    lanes = tuple(
        _parse_lane(item, name=name, level=level, index=index)
        for index, item in enumerate(raw_lanes)
    )
    models = [lane.model for lane in lanes]
    if len(models) != len(set(models)):
        raise _error(f"{where}.lanes contains duplicate models")
    if not any(lane.required for lane in lanes):
        raise _error(f"{where}.lanes must contain at least one required lane")
    judge = (
        DEFAULT_JUDGE_MODEL
        if "judge_model" not in value
        else _exact_slug(value["judge_model"], what=f"{where}.judge_model")
    )
    effort = _effort(value.get("effort", ""), what=f"{where}.effort")
    verify_effort = _effort(value.get("verify_effort", "low"), what=f"{where}.verify_effort")
    tools = _integer(
        value.get("max_tool_turns", 50),
        what=f"{where}.max_tool_turns",
        lower=0,
        upper=1000,
    )
    verify_tools = _integer(
        value.get("verify_max_tool_turns", 30),
        what=f"{where}.verify_max_tool_turns",
        lower=0,
        upper=1000,
    )
    if verify_tools > tools:
        raise _error(f"{where}.verify_max_tool_turns must not exceed max_tool_turns")
    lane_seconds = _integer(
        value.get("lane_timeout_seconds", 600),
        what=f"{where}.lane_timeout_seconds",
        lower=1,
        upper=1800,
    )
    job_seconds = _integer(
        value.get("job_budget_seconds", 1320),
        what=f"{where}.job_budget_seconds",
        lower=240,
        upper=3600,
    )
    if lane_seconds > job_seconds - 180:
        raise _error(f"{where}.lane_timeout_seconds must leave a 180-second publication reserve")
    return _Panel(
        lanes,
        judge,
        effort,
        verify_effort,
        tools,
        verify_tools,
        lane_seconds,
        job_seconds,
    )


def _require_effort_monotonic(name: str, field: str, standard: str, deep: str) -> None:
    where = f"profiles.{name}.deep"
    standard_omitted = standard == ""
    deep_omitted = deep == ""
    if standard_omitted != deep_omitted:
        raise _error(f"{where}.{field} and standard {field} must both be omitted or both explicit")
    if not standard_omitted and _EXPLICIT_EFFORT_RANK[deep] < _EXPLICIT_EFFORT_RANK[standard]:
        raise _error(f"{where} must not lower {field}")


def _require_deep_monotonic(name: str, standard: _Panel, deep: _Panel) -> None:
    required_standard = {lane.model for lane in standard.lanes if lane.required}
    required_deep = {lane.model for lane in deep.lanes if lane.required}
    if not required_standard <= required_deep:
        raise _error(f"profiles.{name}.deep must retain every required standard lane as required")
    if deep.max_tool_turns < standard.max_tool_turns:
        raise _error(f"profiles.{name}.deep must not lower max_tool_turns")
    if deep.verify_max_tool_turns < standard.verify_max_tool_turns:
        raise _error(f"profiles.{name}.deep must not lower verify_max_tool_turns")
    _require_effort_monotonic(name, "effort", standard.effort, deep.effort)
    _require_effort_monotonic(name, "verify_effort", standard.verify_effort, deep.verify_effort)
    if deep.lane_timeout_seconds < standard.lane_timeout_seconds:
        raise _error(f"profiles.{name}.deep must not lower lane_timeout_seconds")
    if deep.job_budget_seconds < standard.job_budget_seconds:
        raise _error(f"profiles.{name}.deep must not lower job_budget_seconds")


def parse_review_profiles(raw: str | None) -> ReviewProfileRegistry | None:
    """Parse the bounded trusted profile registry, or return ``None`` when unset."""
    if raw is None:
        return None
    if type(raw) is not str:
        raise _error("must be a JSON string")
    try:
        raw_bytes = raw.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise _error("is not valid UTF-8") from exc
    if len(raw_bytes) > _MAX_INPUT_BYTES:
        raise _error("exceeds 32 KiB UTF-8 bytes")
    if not raw.strip():
        return None
    value = _parse_json(raw)
    if (
        set(value) != {"version", "profiles"}
        or type(value["version"]) is not int
        or value["version"] != 1
    ):
        raise _error("must be exactly {'version': 1, 'profiles': {...}}")
    profiles = value["profiles"]
    if type(profiles) is not dict or not profiles or len(profiles) > _MAX_PROFILES:
        raise _error(f"profiles must be a non-empty object with at most {_MAX_PROFILES} entries")
    parsed: list[_Profile] = []
    for name, levels in profiles.items():
        if type(name) is not str or not _PROFILE_RE.fullmatch(name):
            raise _error("profile names must match ^[a-z0-9][a-z0-9_-]{0,63}$")
        if type(levels) is not dict or "standard" not in levels or set(levels) - _LEVELS:
            raise _error(f"profiles.{name} must contain standard and optional deep only")
        standard = _parse_panel(levels["standard"], name=name, level="standard")
        deep = None
        if "deep" in levels:
            deep = _parse_panel(levels["deep"], name=name, level="deep")
            _require_deep_monotonic(name, standard, deep)
        parsed.append(_Profile(name, standard, deep))
    return ReviewProfileRegistry(tuple(parsed))


def _validate_routes(routes: Any) -> dict[str, tuple[str | None, str | None]]:
    if type(routes) is not dict:
        raise ActionError("routes must be the normalized parse_model_routes object")
    normalized: dict[str, tuple[str | None, str | None]] = {}
    for model, route in routes.items():
        slug = _exact_slug(model, what="routes key")
        if type(route) is not dict or not route or set(route) - {"provider_order", "service_tier"}:
            raise ActionError("routes entries must be normalized parse_model_routes objects")
        provider: str | None = None
        if "provider_order" in route:
            order = route["provider_order"]
            if (
                type(order) is not list
                or len(order) != 1
                or type(order[0]) is not str
                or not _PROVIDER_RE.fullmatch(order[0])
            ):
                raise ActionError("routes provider_order must contain exactly one provider slug")
            provider = order[0]
        tier: str | None = None
        if "service_tier" in route:
            tier = route["service_tier"]
            if type(tier) is not str or tier not in {"default", "flex", "priority"}:
                raise ActionError("routes service_tier must be default, flex, or priority")
        normalized[slug] = (provider, tier)
    return normalized


def _caller_integer(value: Any, *, what: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ActionError(f"{what} must be an integer from {lower} through {upper}")
    return value


def _canonical_panel(value: _Panel) -> dict[str, Any]:
    return {
        "lanes": [{"model": lane.model, "required": lane.required} for lane in value.lanes],
        "judge_model": value.judge_model,
        "effort": value.effort,
        "verify_effort": value.verify_effort,
        "max_tool_turns": value.max_tool_turns,
        "verify_max_tool_turns": value.verify_max_tool_turns,
        "lane_timeout_seconds": value.lane_timeout_seconds,
        "job_budget_seconds": value.job_budget_seconds,
    }


def _canonical_registry(registry: ReviewProfileRegistry) -> dict[str, Any]:
    return {
        "version": 1,
        "profiles": {
            item.name: {
                "standard": _canonical_panel(item.standard),
                **({"deep": _canonical_panel(item.deep)} if item.deep is not None else {}),
            }
            for item in sorted(registry.profiles, key=lambda item: item.name)
        },
    }


def _digest(
    registry: ReviewProfileRegistry | None,
    routes: dict[str, tuple[str | None, str | None]],
    legacy_panel: _Panel | None = None,
) -> str:
    canonical_registry: Any
    if registry is not None:
        canonical_registry = _canonical_registry(registry)
    else:
        if legacy_panel is None:
            raise AssertionError("legacy digest requires a synthesized panel")
        canonical_registry = {"legacy": {"standard": _canonical_panel(legacy_panel)}}
    payload = {
        "registry": canonical_registry,
        "routes": {
            model: {"provider": provider, "service_tier": tier}
            for model, (provider, tier) in sorted(routes.items())
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def resolve_review_plan(
    registry: ReviewProfileRegistry | None,
    *,
    profile: str,
    minimum: str,
    requested_level: str,
    mode: str,
    models: list[str],
    judge_model: str,
    routes: dict,
    effort: str,
    max_tool_turns: int,
    job_budget_seconds: int,
    lane_timeout_seconds: int = 600,
) -> ReviewPlan:
    """Resolve one immutable execution plan without I/O or caller-side mutation."""
    if registry is not None and type(registry) is not ReviewProfileRegistry:
        raise ActionError("registry must be returned by parse_review_profiles")
    if type(profile) is not str or not _PROFILE_RE.fullmatch(profile):
        raise ActionError("profile must match ^[a-z0-9][a-z0-9_-]{0,63}$")
    if type(minimum) is not str or minimum not in _LEVELS:
        raise ActionError("minimum must be standard or deep")
    if type(requested_level) is not str or requested_level not in _REQUESTED_LEVELS:
        raise ActionError("requested_level must be auto or deep")
    if type(mode) is not str or mode not in _MODES:
        raise ActionError("mode must be initial or verify")
    caller_tools = _caller_integer(max_tool_turns, what="max_tool_turns", lower=0, upper=1000)
    caller_job = _caller_integer(
        job_budget_seconds, what="job_budget_seconds", lower=240, upper=3600
    )
    caller_lane = _caller_integer(
        lane_timeout_seconds, what="lane_timeout_seconds", lower=1, upper=1800
    )
    bound_routes = _validate_routes(routes)

    level = "deep" if minimum == "deep" or requested_level == "deep" else "standard"
    trigger = (
        "manual" if requested_level == "deep" else "policy" if minimum == "deep" else "baseline"
    )
    legacy = registry is None
    if registry is None:
        if profile not in {"code", "docs"}:
            raise ActionError(f"legacy configuration does not define profile {profile!r}")
        if level == "deep":
            raise ActionError(f"profile {profile!r} has no deep mapping")
        if type(models) is not list or not models or len(models) > _MAX_LANES:
            raise ActionError("models must contain 1 through 4 lanes")
        panel = _Panel(
            tuple(ReviewLane(_exact_slug(model, what="model"), False) for model in models),
            _exact_slug(judge_model, what="judge_model"),
            _effort(effort, what="effort"),
            _effort(effort, what="effort"),
            caller_tools,
            caller_tools,
            caller_lane,
            caller_job,
        )
    else:
        selected = registry.profile(profile)
        if selected is None:
            raise ActionError(f"review profile {profile!r} is not configured")
        panel = selected.deep if level == "deep" else selected.standard
        if panel is None:
            raise ActionError(f"profile {profile!r} has no deep mapping")

    selected_tools = panel.max_tool_turns if mode == "initial" else panel.verify_max_tool_turns
    selected_effort = panel.effort if mode == "initial" else panel.verify_effort
    if selected_tools > caller_tools:
        raise ActionError("selected profile max_tool_turns exceeds caller ceiling")
    if panel.job_budget_seconds > caller_job:
        raise ActionError("selected profile job_budget_seconds exceeds caller ceiling")
    if panel.lane_timeout_seconds > caller_lane:
        raise ActionError("selected profile lane_timeout_seconds exceeds caller ceiling")
    lanes = tuple(
        ReviewLane(lane.model, lane.required, *bound_routes.get(lane.model, (None, None)))
        for lane in panel.lanes
    )
    return ReviewPlan(
        profile,
        level,
        trigger,
        lanes,
        panel.judge_model,
        selected_effort,
        selected_tools,
        panel.lane_timeout_seconds,
        panel.job_budget_seconds,
        _digest(registry, bound_routes, panel if legacy else None),
        legacy,
    )
