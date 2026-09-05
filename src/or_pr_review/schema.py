"""Structured findings and the lane-artifact contract.

Model output that does not parse is a lane failure (fail-open).
A lane artifact that does not match this schema fails the job (fail-closed).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any

from or_pr_review.errors import LaneError, SchemaError

SCHEMA_VERSION = 1
SEVERITIES = ("bug", "risk", "nit")
# Severity order is a review-wide policy: de-duplication, ledger trimming,
# and capped judge output must all prefer the same findings.
SEVERITY_RANK = {"bug": 2, "risk": 1, "nit": 0}
RESOLUTION_STATUSES = ("fixed", "not_fixed", "fixed_incorrectly", "disputed")
MAX_TITLE = 200
MAX_BODY = 8_000
MAX_FILE = 500
MAX_FINDINGS = 80
MAX_COVERAGE_ENTRIES = 500
MAX_RESOLUTIONS = 200
MAX_RESOLUTION_NOTE = 2_000
MAX_FINDING_ID = 32

_FINDINGS_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "maxItems": MAX_FINDINGS,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "severity": {"type": "string", "enum": list(SEVERITIES)},
                    "file": {"type": ["string", "null"]},
                    "line": {"type": ["integer", "null"]},
                },
                "required": ["title", "body", "severity", "file", "line"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}


def findings_json_schema(
    *, include_coverage: bool = False, include_resolutions: bool = False
) -> dict[str, Any]:
    """OpenRouter/OpenAI response_format json_schema payload.

    Initial rounds require a per-file coverage manifest; verify rounds
    require a resolution entry per carried finding.
    """
    schema = json.loads(json.dumps(_FINDINGS_RESPONSE_SCHEMA))
    if include_coverage:
        schema["properties"]["coverage"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "findings": {"type": "integer"},
                },
                "required": ["path", "findings"],
                "additionalProperties": False,
            },
        }
        schema["required"] = [*schema["required"], "coverage"]
    if include_resolutions:
        schema["properties"]["resolutions"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": list(RESOLUTION_STATUSES),
                        "description": (
                            "Authoritative disposition. fixed means fully fixed; not_fixed "
                            "means the original issue remains; fixed_incorrectly means an "
                            "attempted fix is wrong or incomplete; disputed means the original "
                            "finding is invalid or intentionally accepted."
                        ),
                    },
                    "note": {
                        "type": "string",
                        "description": (
                            "Evidence for the selected status. It must agree with status and "
                            "must not state a different disposition."
                        ),
                    },
                },
                "required": ["id", "status", "note"],
                "additionalProperties": False,
            },
        }
        schema["required"] = [*schema["required"], "resolutions"]
    return {
        "name": "pr_review_findings",
        "strict": True,
        "schema": schema,
    }


@dataclass(frozen=True)
class Finding:
    title: str
    body: str
    severity: str
    file: str | None
    line: int | None
    model_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Resolution:
    """The model's verdict on one carried finding during a verify round."""

    id: str
    status: str  # fixed | not_fixed | fixed_incorrectly | disputed
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LaneResult:
    schema_version: int
    ok: bool
    model: str
    findings: list[Finding] = field(default_factory=list)
    error: str | None = None
    elapsed_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    requests: int | None = None
    tool_rounds: int | None = None
    retries: int | None = None
    salvaged: bool = False
    head_sha: str | None = None
    # The upstream provider OpenRouter routed to (last response wins) — a
    # model slug can be served by several providers with different behavior.
    provider: str | None = None
    resolutions: list[Resolution] = field(default_factory=list)
    coverage: list[tuple[str, int]] = field(default_factory=list)
    # Aggregate-safe Gemini continuity diagnostics. No signature values or
    # transcript contents are persisted. Keep new fields at the end so older
    # positional LaneResult construction remains source-compatible.
    thought_signature_tool_turns: int | None = None
    thought_signature_recoveries: int | None = None
    sanitized_tool_turns: int | None = None
    dropped_findings: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ok": self.ok,
            "model": self.model,
            "findings": [finding.to_dict() for finding in self.findings],
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "cost_usd": self.cost_usd,
            "requests": self.requests,
            "tool_rounds": self.tool_rounds,
            "retries": self.retries,
            "salvaged": self.salvaged,
            "thought_signature_tool_turns": self.thought_signature_tool_turns,
            "thought_signature_recoveries": self.thought_signature_recoveries,
            "sanitized_tool_turns": self.sanitized_tool_turns,
            "dropped_findings": self.dropped_findings,
            "head_sha": self.head_sha,
            "provider": self.provider,
            "resolutions": [resolution.to_dict() for resolution in self.resolutions],
            "coverage": [{"path": path, "findings": count} for path, count in self.coverage],
        }


def normalize_review_path(value: str) -> str | None:
    """A safe repository-relative path: no traversal, backslashes, backticks,
    control characters, or absolute paths.

    Returning one canonical POSIX spelling prevents a lane from making the
    same reviewed file look like multiple paths (for example ``a//b.py``).
    This is defense in depth: callers still validate locations against the
    reviewed diff before publishing an inline comment.
    """
    path = value.strip()
    if (
        not path
        or len(path) > MAX_FILE
        or "\\" in path
        or "`" in path
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
    ):
        return None
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        return None
    normalized = parsed.as_posix()
    return normalized if normalized and len(normalized) <= MAX_FILE else None


def valid_review_path(value: str) -> bool:
    """Compatibility predicate for callers that only need path validity."""
    return normalize_review_path(value) is not None


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    raise LaneError("finding file must be a string or null")


def _as_optional_line(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise LaneError("finding line must be an integer or null")
    if isinstance(value, int):
        if value <= 0:
            return None
        return value
    if isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
        return number if number > 0 else None
    raise LaneError("finding line must be an integer or null")


def parse_finding(raw: object, model_id: str) -> Finding:
    if not isinstance(raw, dict):
        raise LaneError("each finding must be an object")
    title = raw.get("title")
    body = raw.get("body")
    severity = raw.get("severity")
    if not isinstance(title, str) or not title.strip():
        raise LaneError("finding title is required")
    if not isinstance(body, str) or not body.strip():
        raise LaneError("finding body is required")
    if not isinstance(severity, str) or severity.strip().lower() not in SEVERITIES:
        raise LaneError(f"finding severity must be one of {', '.join(SEVERITIES)}")
    path = _as_optional_str(raw.get("file") if "file" in raw else raw.get("path"))
    if path:
        # Fail-open: keep the finding, drop the unsafe path (traversal,
        # backticks, control characters, absolute or oversized paths).
        path = normalize_review_path(path)
    return Finding(
        title=title.strip()[:MAX_TITLE],
        body=body.strip()[:MAX_BODY],
        severity=severity.strip().lower(),
        file=path,
        line=_as_optional_line(raw.get("line")),
        model_id=model_id,
    )


def parse_model_findings(payload: object, model_id: str) -> list[Finding]:
    """Parse a model's structured review. Raises LaneError on mismatch."""
    return parse_lane_payload(payload, model_id)[0]


def parse_lane_payload(
    payload: object,
    model_id: str,
    *,
    expect_coverage: bool = False,
    expect_resolutions: bool = False,
    diagnostics: dict[str, int] | None = None,
) -> tuple[list[Finding], list[Resolution], list[tuple[str, int]]]:
    """Parse findings plus the optional coverage manifest and resolutions.

    Raises LaneError on mismatch (the lane fails open). `expect_*` makes the
    corresponding section mandatory: initial rounds must account for every
    embedded-diff file; verify rounds must resolve every carried finding.
    """
    if isinstance(payload, str):
        payload = extract_json_object(payload)
    if not isinstance(payload, dict):
        raise LaneError("model output must be a JSON object")
    findings_raw = payload.get("findings")
    if findings_raw is None:
        raise LaneError("model output is missing a findings array")
    if not isinstance(findings_raw, list):
        raise LaneError("findings must be an array")
    findings = [parse_finding(item, model_id) for item in findings_raw]
    dropped = max(0, len(findings) - MAX_FINDINGS)
    if dropped:
        # The response_format schema advertises maxItems, so only the
        # schema-free tool path can get here. Say so instead of silently
        # narrowing an "exhaustive" review.
        print(
            f"warning: model returned {len(findings_raw)} findings; keeping "
            f"the strongest {MAX_FINDINGS} and omitting {dropped}"
        )
        findings = sorted(findings, key=lambda finding: -SEVERITY_RANK[finding.severity])[
            :MAX_FINDINGS
        ]
    coverage = _parse_coverage(payload.get("coverage"), required=expect_coverage)
    resolutions = _parse_resolutions(payload.get("resolutions"), required=expect_resolutions)
    if diagnostics is not None:
        diagnostics["dropped_findings"] = dropped
    return findings, resolutions, coverage


def _parse_coverage(raw: object, *, required: bool) -> list[tuple[str, int]]:
    if raw is None:
        if required:
            raise LaneError(
                "coverage is missing; the manifest must account for every embedded-diff file"
            )
        return []
    if not isinstance(raw, list):
        raise LaneError("coverage must be an array or absent")
    entries: list[tuple[str, int]] = []
    positions: dict[str, int] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise LaneError(f"coverage[{index}] must be an object")
        path = item.get("path")
        count = item.get("findings")
        if not isinstance(path, str) or not valid_review_path(path):
            raise LaneError(f"coverage[{index}].path must be a safe relative path")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise LaneError(f"coverage[{index}].findings must be a nonnegative integer")
        normalized = path.strip()
        if normalized in positions:
            existing_index = positions[normalized]
            existing_path, existing_count = entries[existing_index]
            # Repeated model-authored rows describe the same file. Keep the
            # strongest claim without double-counting the manifest.
            entries[existing_index] = (existing_path, max(existing_count, count))
            continue
        positions[normalized] = len(entries)
        entries.append((normalized, count))
    if len(entries) > MAX_COVERAGE_ENTRIES:
        if required:
            raise LaneError(f"coverage exceeds the limit of {MAX_COVERAGE_ENTRIES}")
        # Enforcement is off (diff too large for a manifest); keep what fits
        # instead of failing the lane over advisory extra entries.
        entries = entries[:MAX_COVERAGE_ENTRIES]
    return entries


def _parse_resolutions(raw: object, *, required: bool) -> list[Resolution]:
    if raw is None:
        if required:
            raise LaneError(
                "resolutions are missing; every carried finding needs a resolution entry"
            )
        return []
    if not isinstance(raw, list):
        raise LaneError("resolutions must be an array or absent")
    if len(raw) > MAX_RESOLUTIONS:
        raise LaneError(f"resolutions exceeds the limit of {MAX_RESOLUTIONS}")
    resolutions: list[Resolution] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise LaneError(f"resolutions[{index}] must be an object")
        ident = item.get("id")
        if not isinstance(ident, str) or not ident.strip() or len(ident.strip()) > MAX_FINDING_ID:
            raise LaneError(f"resolutions[{index}].id must be a short non-empty string")
        status_value = item.get("status")
        if not isinstance(status_value, str):
            raise LaneError(f"resolutions[{index}].status must be a string")
        status = status_value.strip().lower().replace("-", "_").replace(" ", "_")
        if status not in RESOLUTION_STATUSES:
            raise LaneError(f"resolutions[{index}].status is invalid")
        note = item.get("note")
        if note is not None and not isinstance(note, str):
            raise LaneError(f"resolutions[{index}].note must be a string or null")
        try:
            note_text = validate_resolution_note(status, (note or "").strip())
        except LaneError as exc:
            raise LaneError(
                f"resolutions[{index}] for {ident.strip()!r} is inconsistent: {exc}"
            ) from exc
        resolutions.append(Resolution(id=ident.strip(), status=status, note=note_text))
    return resolutions


_LEADING_DISPOSITION_RE = re.compile(
    r"^(?:actually\s+)?(?:(?P<prefix>(?:(?:the\s+)?(?:finding|issue|bug|risk|problem|"
    r"fix|change|implementation)|it|this)\s+(?:is|was|remains)\s+|"
    r"(?:status|verdict|resolution)\s*:\s*))?[\s`*_~-]*"
    r"(?P<disposition>fixed[\s_-]+incorrectly|fixed[\s_-]+correctly|not[\s_-]+fixed|"
    r"disputed|fixed)\b",
    re.IGNORECASE,
)
_DISPOSITION_ALIASES = {
    "fixed": "fixed",
    "fixed correctly": "fixed",
    "not fixed": "not_fixed",
    "fixed incorrectly": "fixed_incorrectly",
    "disputed": "disputed",
}


def validate_resolution_note(status: str, note: str) -> str:
    """Reject an explicit prose disposition that disagrees with ``status``.

    Models occasionally emit an authoritative enum and then start the note
    with a different verdict (for example, ``fixed_incorrectly`` plus
    "Actually fixed correctly"). Silently choosing either half can keep a
    fixed finding open or close a genuinely open one. Only an explicit leading
    disposition is inspected; a conflict fails validation so the harness can
    request one schema-enforced correction without inferring either verdict.
    """
    text = note.strip()[:MAX_RESOLUTION_NOTE]
    if not text:
        return ""
    match = _LEADING_DISPOSITION_RE.match(text)
    if match is None:
        return text
    declared = match.group("disposition").lower().replace("_", " ").replace("-", " ")
    declared = re.sub(r"\s+", " ", declared).strip()
    # A leading verb such as "Fixed the unit tests, but the production race
    # remains" is supporting evidence, not a second structured disposition.
    # Bare ``fixed`` is authoritative only in an explicit subject/status form;
    # the qualified canonical phrases remain unambiguous without a prefix.
    if declared == "fixed" and match.group("prefix") is None:
        return text
    if _DISPOSITION_ALIASES.get(declared) == status:
        return text
    raise LaneError(
        "resolution note contradicts its authoritative structured status: "
        f"status is {status!r}, but the note declares {declared!r}"
    )


def validate_coverage(coverage: list[tuple[str, int]], expected_paths: set[str]) -> str | None:
    """Reject a manifest that does not account for the embedded diff.

    Coverage is required only for files in the embedded diff; findings on
    other paths are valid blast-radius results and never checked here.
    """
    if not expected_paths:
        return None
    covered = {path for path, _count in coverage}
    if not covered:
        return "coverage is missing; the manifest must account for every embedded-diff file"
    missing = sorted(expected_paths - covered)
    if missing:
        named = ", ".join(missing[:5])
        return f"coverage does not account for {len(missing)} diff file(s): {named}"
    extra = sorted(covered - expected_paths)
    if extra:
        named = ", ".join(extra[:5])
        return f"coverage lists file(s) not in the embedded diff: {named}"
    return None


def coverage_count_mismatches(
    findings: list[Finding],
    coverage: list[tuple[str, int]],
    expected_paths: set[str],
) -> list[str]:
    """Per-file counts that disagree with reported in-diff findings.

    Not fatal: the recovered findings are kept and the review posts a
    completed verdict with a visible note (sibling-Grok 1.0.5 behavior).
    """
    reported: dict[str, int] = {}
    for finding in findings:
        if finding.file and finding.file in expected_paths:
            reported[finding.file] = reported.get(finding.file, 0) + 1
    notes: list[str] = []
    for path, count in coverage:
        if path not in expected_paths:
            continue
        actual = reported.get(path, 0)
        if actual != count:
            notes.append(
                f"coverage claims {count} finding(s) in {path!r} but {actual} were reported"
            )
    return notes


def extract_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise LaneError("model returned empty text")
    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*\n(.*)\n```\s*$", stripped, re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise LaneError("model returned no JSON object")
    try:
        parsed = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LaneError(f"model JSON did not parse: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LaneError("model JSON must be an object")
    return parsed


def parse_lane_artifact(payload: object) -> LaneResult:
    """Parse a lane JSON file. Schema mismatches fail-closed."""
    if not isinstance(payload, dict):
        raise SchemaError("lane artifact must be a JSON object")
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SchemaError(f"lane artifact schema_version must be {SCHEMA_VERSION}, got {version!r}")
    missing = {"ok", "model", "findings", "error"} - set(payload)
    if missing:
        raise SchemaError(f"lane artifact missing required keys: {sorted(missing)}")
    ok = payload.get("ok")
    model = payload.get("model")
    error = payload.get("error")
    findings_raw = payload.get("findings")
    if not isinstance(ok, bool):
        raise SchemaError("lane artifact ok must be a boolean")
    if not isinstance(model, str) or not model.strip():
        raise SchemaError("lane artifact model must be a non-empty string")
    if error is not None and not isinstance(error, str):
        raise SchemaError("lane artifact error must be a string or null")
    if not isinstance(findings_raw, list):
        raise SchemaError("lane artifact findings must be an array")
    findings: list[Finding] = []
    for item in findings_raw:
        try:
            findings.append(parse_finding(item, model))
        except LaneError as exc:
            raise SchemaError(f"lane artifact finding is invalid: {exc}") from exc
    salvaged = payload.get("salvaged", False)
    dropped = payload.get("dropped_findings", 0)
    if isinstance(dropped, bool) or not isinstance(dropped, int) or dropped < 0:
        raise SchemaError("lane artifact dropped_findings must be a nonnegative integer")
    if not isinstance(salvaged, bool):
        raise SchemaError("lane artifact salvaged must be a boolean")
    head_sha_value = payload.get("head_sha")
    if head_sha_value is not None and not isinstance(head_sha_value, str):
        raise SchemaError("lane artifact head_sha must be a string or null")
    head_sha = (
        head_sha_value.strip().lower()
        if isinstance(head_sha_value, str) and head_sha_value.strip()
        else None
    )
    provider_value = payload.get("provider")
    if provider_value is not None and not isinstance(provider_value, str):
        raise SchemaError("lane artifact provider must be a string or null")
    provider = (
        provider_value.strip()[:100]
        if isinstance(provider_value, str) and provider_value.strip()
        else None
    )
    try:
        resolutions = _parse_resolutions(payload.get("resolutions"), required=False)
        coverage = _parse_coverage(payload.get("coverage"), required=False)
    except LaneError as exc:
        raise SchemaError(f"lane artifact is invalid: {exc}") from exc
    return LaneResult(
        schema_version=SCHEMA_VERSION,
        ok=ok,
        model=model.strip(),
        findings=findings,
        error=error,
        elapsed_ms=_optional_int(payload.get("elapsed_ms"), field="elapsed_ms"),
        prompt_tokens=_optional_int(payload.get("prompt_tokens"), field="prompt_tokens"),
        completion_tokens=_optional_int(
            payload.get("completion_tokens"), field="completion_tokens"
        ),
        cached_tokens=_optional_int(payload.get("cached_tokens"), field="cached_tokens"),
        cost_usd=_optional_float(payload.get("cost_usd"), field="cost_usd"),
        requests=_optional_int(payload.get("requests"), field="requests"),
        tool_rounds=_optional_int(payload.get("tool_rounds"), field="tool_rounds"),
        retries=_optional_int(payload.get("retries"), field="retries"),
        salvaged=salvaged,
        thought_signature_tool_turns=_optional_int(
            payload.get("thought_signature_tool_turns"), field="thought_signature_tool_turns"
        ),
        thought_signature_recoveries=_optional_int(
            payload.get("thought_signature_recoveries"), field="thought_signature_recoveries"
        ),
        sanitized_tool_turns=_optional_int(
            payload.get("sanitized_tool_turns"), field="sanitized_tool_turns"
        ),
        head_sha=head_sha,
        provider=provider,
        resolutions=resolutions,
        coverage=coverage,
        dropped_findings=dropped,
    )


def _optional_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"lane artifact {field} must be an integer or null")
    return value


def _optional_float(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"lane artifact {field} must be a number or null")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise SchemaError(f"lane artifact {field} must be finite and non-negative")
    return converted


def failed_lane(model: str, error: str, elapsed_ms: int | None = None) -> LaneResult:
    return LaneResult(
        schema_version=SCHEMA_VERSION,
        ok=False,
        model=model,
        findings=[],
        error=error,
        elapsed_ms=elapsed_ms,
    )
