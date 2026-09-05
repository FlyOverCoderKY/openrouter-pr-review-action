"""Versioned publication snapshots carried by matrix lane artifacts.

The digest detects inconsistent artifacts; it is not a signature. These files
must come from the trusted lane jobs of the same workflow run.
"""

from __future__ import annotations

import hashlib
import json
import re
import types
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from or_pr_review.collect import CollectedReview
from or_pr_review.errors import SchemaError
from or_pr_review.loop import LoopState
from or_pr_review.schema import SEVERITIES, valid_review_path

MAX_CONTEXT_BYTES = 16 * 1024 * 1024
CONTEXT_VERSION = 1


@dataclass(frozen=True)
class ReviewContext:
    repository: str
    collected: CollectedReview
    loop: LoopState
    max_tool_turns: int


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _decode(kind: Any, value: Any) -> Any:
    """Restore only the statically declared dataclasses and their primitive fields."""
    origin, args = get_origin(kind), get_args(kind)
    if origin in (Union, types.UnionType):
        for choice in args:
            try:
                return _decode(choice, value)
            except SchemaError:
                pass
    elif origin is Literal:
        if type(value) is str and value in args:
            return value
    elif origin is tuple:
        if isinstance(value, list):
            return tuple(_decode(args[0], item) for item in value)
    elif is_dataclass(kind):
        if isinstance(value, dict) and set(value) == {field.name for field in fields(kind)}:
            hints = get_type_hints(kind)
            return kind(**{name: _decode(hints[name], item) for name, item in value.items()})
    elif kind in (str, int, bool, type(None)) and type(value) is kind:
        return value
    raise SchemaError("review context has an invalid field or shape")


def freeze_context(
    repository: str, collected: CollectedReview, loop: LoopState, max_tool_turns: int
) -> dict[str, Any]:
    payload = asdict(ReviewContext(repository, collected, loop, max_tool_turns))
    # Normalize tuples exactly as the artifact writer will. Validate before any
    # paid request so an oversized/invalid context cannot strand completed work.
    payload = json.loads(_canonical(payload))
    envelope = {
        "version": CONTEXT_VERSION,
        "sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        "payload": payload,
    }
    restore_context(envelope)
    return envelope


def restore_context(envelope: object) -> ReviewContext:
    if not isinstance(envelope, dict) or set(envelope) != {"version", "sha256", "payload"}:
        raise SchemaError("matrix lane is missing a valid publication context; rerun its lanes")
    if type(envelope["version"]) is not int or envelope["version"] != CONTEXT_VERSION:
        raise SchemaError("unsupported review context version; rerun the lanes with this action")
    try:
        encoded = _canonical(envelope)
        digest = hashlib.sha256(_canonical(envelope["payload"])).hexdigest()
    except (ValueError, TypeError, RecursionError) as exc:
        raise SchemaError("review context is not bounded JSON") from exc
    if len(encoded) > MAX_CONTEXT_BYTES:
        raise SchemaError("review context exceeds the 16 MiB artifact limit")
    if envelope["sha256"] != digest:
        raise SchemaError("review context digest mismatch")
    context = _decode(ReviewContext, envelope["payload"])
    collected, loop = context.collected, context.loop
    if (
        not re.fullmatch(r"[^/\s]+/[^/\s]+", context.repository)
        or collected.pr_number < 1
        or not re.fullmatch(r"[0-9a-f]{40}", collected.head_sha)
        or collected.plan.to_sha != collected.head_sha
        or loop.mode != collected.mode
        or loop.mode not in {"initial", "verify"}
        or not 1 <= loop.round_number <= 999
        or not re.fullmatch(r"(?:[0-9a-f]{12})?", loop.generation)
        or not 0 <= context.max_tool_turns <= 1000
        or collected.truncation.max_diff_kb < 1
        or collected.truncation.original_bytes < 0
        or collected.truncation.embedded_bytes != len(collected.diff.encode("utf-8"))
        or collected.truncation.original_bytes < collected.truncation.embedded_bytes
    ):
        raise SchemaError("review context contains inconsistent publication metadata")
    findings = (*loop.prior_findings, *loop.retired_prior)
    if len(findings) > 200 or len({item.id for item in findings}) != len(findings):
        raise SchemaError("review context has invalid carried finding identities")
    for item in findings:
        if (
            not re.fullmatch(r"r\d{1,3}-\d{1,3}", item.id)
            or item.severity not in SEVERITIES
            or item.status not in {"open", "disputed"}
            or (item.file is not None and not valid_review_path(item.file))
            or (item.line is not None and item.line < 1)
            or len(item.title) > 300
            or len(item.evidence) > 616
            or len(item.models) > 8
        ):
            raise SchemaError("review context has an invalid carried finding")
    return context
