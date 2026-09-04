"""Parse the comma-separated OpenRouter models list and enforce the lane cap."""

from __future__ import annotations

import json
import re

from or_pr_review.errors import ActionError

LANE_CAP = 4
DEFAULT_MODEL = "x-ai/grok-4.6"
# Verified live on OpenRouter 2026-08-29. Merge/de-dupe only; not a second reviewer.
DEFAULT_JUDGE_MODEL = "openai/gpt-5.6-luna"

# OpenRouter slugs look like provider/model or provider/model:variant.
# Do not invent slugs; callers pass catalogue ids.
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*(:[A-Za-z0-9._-]+)?$")


def parse_slug(raw: str, *, what: str) -> str:
    slug = raw.strip()
    if not _SLUG_RE.fullmatch(slug):
        raise ActionError(
            f"invalid OpenRouter {what} slug {slug!r}; "
            "expected provider/model (optional :variant), e.g. google/gemini-3.1-flash-lite"
        )
    return slug


def parse_judge_model(raw: str | None, *, default: str = DEFAULT_JUDGE_MODEL) -> str:
    text = (raw or "").strip()
    return parse_slug(text or default, what="judge_model")


def judge_is_needed(slugs: list[str]) -> bool:
    """One configured review lane (or a future single-persona run) skips the judge."""
    return len(slugs) >= 2


def parse_models(raw: str | None, *, default: str = DEFAULT_MODEL) -> list[str]:
    """Split a comma-separated models list. Empty input uses the default slug."""
    text = (raw or "").strip()
    if not text:
        slugs = [default]
    else:
        slugs = [part.strip() for part in text.split(",") if part.strip()]
    if not slugs:
        raise ActionError("models is empty after parsing; provide at least one OpenRouter slug")
    if len(slugs) > LANE_CAP:
        raise ActionError(
            f"models lists {len(slugs)} lanes; the hard cap is {LANE_CAP}. "
            f"Shorten the list to {LANE_CAP} or fewer OpenRouter slugs."
        )
    return [parse_slug(slug, what="model") for slug in slugs]


def matrix_payload(slugs: list[str]) -> list[dict[str, object]]:
    """GitHub Actions matrix.include entries (index + model)."""
    return [{"index": index, "model": slug} for index, slug in enumerate(slugs)]


def models_json(slugs: list[str]) -> str:
    return json.dumps(slugs)


def matrix_json(slugs: list[str]) -> str:
    return json.dumps(matrix_payload(slugs))
