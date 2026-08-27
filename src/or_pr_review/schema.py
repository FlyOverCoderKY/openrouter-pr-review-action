"""Structured findings and the lane-artifact contract.

Model output that does not parse is a lane failure (fail-open).
A lane artifact that does not match this schema fails the job (fail-closed).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any

from or_pr_review.errors import LaneError, SchemaError

SCHEMA_VERSION = 1
SEVERITIES = ("bug", "risk", "nit")
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
                    "status": {"type": "string", "enum": list(RESOLUTION_STATUSES)},
                    "note": {"type": "string"},
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
    requests: int | None = None
    tool_rounds: int | None = None
    retries: int | None = None
    salvaged: bool = False
    head_sha: str | None = None
    resolutions: list[Resolution] = field(default_factory=list)
    coverage: list[tuple[str, int]] = field(default_factory=list)

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
            "requests": self.requests,
            "tool_rounds": self.tool_rounds,
            "retries": self.retries,
            "salvaged": self.salvaged,
            "head_sha": self.head_sha,
            "resolutions": [resolution.to_dict() for resolution in self.resolutions],
            "coverage": [
                {"path": path, "findings": count} for path, count in self.coverage
            ],
        }


def valid_review_path(value: str) -> bool:
    """A safe repository-relative path: no traversal, backslashes, backticks,
    control characters, or absolute paths. Mirrors the sibling Grok harness."""
    path = value.strip()
    if (
        not path
        or len(path) > MAX_FILE
        or "\\" in path
        or "`" in path
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
    ):
        return False
    parsed = PurePosixPath(path)
    return not parsed.is_absolute() and all(part not in {"", ".", ".."} for part in parsed.parts)


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
    if path and not valid_review_path(path):
        # Fail-open: keep the finding, drop the unsafe path (traversal,
        # backticks, control characters, absolute or oversized paths).
        path = None
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
    if len(findings_raw) > MAX_FINDINGS:
        # The response_format schema advertises maxItems, so only the
        # schema-free tool path can get here. Say so instead of silently
        # narrowing an "exhaustive" review.
        print(
            f"warning: model returned {len(findings_raw)} findings; keeping "
            f"the first {MAX_FINDINGS}"
        )
        findings_raw = findings_raw[:MAX_FINDINGS]
    findings = [parse_finding(item, model_id) for item in findings_raw]
    coverage = _parse_coverage(payload.get("coverage"), required=expect_coverage)
    resolutions = _parse_resolutions(payload.get("resolutions"), required=expect_resolutions)
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
    if len(raw) > MAX_COVERAGE_ENTRIES:
        if required:
            raise LaneError(f"coverage exceeds the limit of {MAX_COVERAGE_ENTRIES}")
        # Enforcement is off (diff too large for a manifest); keep what fits
        # instead of failing the lane over advisory extra entries.
        raw = raw[:MAX_COVERAGE_ENTRIES]
    entries: list[tuple[str, int]] = []
    seen: set[str] = set()
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
        if normalized in seen:
            raise LaneError(f"coverage lists {normalized!r} more than once")
        seen.add(normalized)
        entries.append((normalized, count))
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
        note_text = (note or "").strip()[:MAX_RESOLUTION_NOTE]
        resolutions.append(Resolution(id=ident.strip(), status=status, note=note_text))
    return resolutions


def validate_coverage(
    coverage: list[tuple[str, int]], expected_paths: set[str]
) -> str | None:
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
        raise SchemaError(
            f"lane artifact schema_version must be {SCHEMA_VERSION}, got {version!r}"
        )
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
        elapsed_ms=_optional_int(payload.get("elapsed_ms")),
        prompt_tokens=_optional_int(payload.get("prompt_tokens")),
        completion_tokens=_optional_int(payload.get("completion_tokens")),
        cached_tokens=_optional_int(payload.get("cached_tokens")),
        requests=_optional_int(payload.get("requests")),
        tool_rounds=_optional_int(payload.get("tool_rounds")),
        retries=_optional_int(payload.get("retries")),
        salvaged=salvaged,
        head_sha=head_sha,
        resolutions=resolutions,
        coverage=coverage,
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError("numeric lane fields must be integers or null")
    return value


def failed_lane(model: str, error: str, elapsed_ms: int | None = None) -> LaneResult:
    return LaneResult(
        schema_version=SCHEMA_VERSION,
        ok=False,
        model=model,
        findings=[],
        error=error,
        elapsed_ms=elapsed_ms,
    )
