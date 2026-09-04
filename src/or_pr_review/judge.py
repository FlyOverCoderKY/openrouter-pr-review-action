"""OpenRouter judge: recall-safe union-merge of structured lane findings.

The judge is not a second reviewer and not a filter. It is skipped when only
one review lane is configured. Recall safety is enforced by IDENTITY, not
counts: every input finding carries a source id, every output issue must
name the source ids it merged, and verification repairs any input finding
the judge failed to account for by appending it verbatim. A judge output
whose accounting cannot be trusted (unknown ids, missing sources) is
replaced wholesale by a deterministic union that cannot lose a finding.
Judge transport or structural schema failures are caught by the orchestrator
and produce a visibly labeled deterministic union; invalid lane artifacts and
other action-wide contract failures remain fail-closed.
"""

from __future__ import annotations

import json
from typing import Any

from or_pr_review.errors import ActionError, SchemaError
from or_pr_review.harness import (
    MAX_GEMINI_RESPONSE_TOKENS,
    ChatFn,
    _is_gemini_model,
    _response_spend,
    openrouter_chat,
    response_message_text,
)
from or_pr_review.merge import (
    MergedIssue,
    absorb_merged_issue,
    deduplicate_issues,
    is_environmental_diagnostic,
    same_merged_issue,
)
from or_pr_review.redaction import redact
from or_pr_review.schema import (
    MAX_BODY,
    MAX_FINDINGS,
    MAX_TITLE,
    SEVERITIES,
    extract_json_object,
    valid_review_path,
)

# Keep the judge clerical and low-latency. Luna's adoption benchmark used this setting.
JUDGE_REASONING = {"effort": "minimal"}

_SEVERITY_RANK = {"bug": 2, "risk": 1, "nit": 0}


def judge_json_schema() -> dict[str, Any]:
    return {
        "name": "merged_pr_review_issues",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                # No maxItems here: Gemini's structured-output subset rejects
                # it (INVALID_ARGUMENT). The parse side still caps at
                # MAX_FINDINGS fail-closed.
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "body": {"type": "string"},
                            "severity": {"type": "string", "enum": list(SEVERITIES)},
                            "file": {"type": ["string", "null"]},
                            "line": {"type": ["integer", "null"]},
                            "models": {"type": "array", "items": {"type": "string"}},
                            "sources": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": [
                            "title",
                            "body",
                            "severity",
                            "file",
                            "line",
                            "models",
                            "sources",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["issues"],
            "additionalProperties": False,
        },
    }


def _finding_id(lane_index: int, finding_index: int) -> str:
    return f"{lane_index}.{finding_index}"


def _annotated_lanes(lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated = []
    for lane_index, lane in enumerate(lanes):
        findings = []
        for finding_index, finding in enumerate(lane.get("findings") or []):
            if isinstance(finding, dict):
                findings.append({"id": _finding_id(lane_index, finding_index), **finding})
        annotated.append({"model": lane.get("model"), "findings": findings})
    return annotated


def partition_reviewable_lanes(
    lanes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Separate code findings from temporary review-environment failures.

    Returns shallow lane copies plus ``(model, title)`` diagnostics. Keeping
    this deterministic and side-effect free makes it reusable by local judge
    benchmarks as well as the production path.
    """
    reviewable: list[dict[str, Any]] = []
    diagnostics: list[tuple[str, str]] = []
    for lane in lanes:
        model = str(lane.get("model") or "")
        findings = []
        for finding in lane.get("findings") or []:
            if not isinstance(finding, dict):
                continue
            title = str(finding.get("title") or "").strip()
            body = str(finding.get("body") or "").strip()
            line = finding.get("line")
            line_n = line if isinstance(line, int) and not isinstance(line, bool) else None
            if is_environmental_diagnostic(title=title, body=body, line=line_n):
                diagnostics.append((model, title))
                continue
            findings.append(finding)
        reviewable.append({**lane, "findings": findings})
    return reviewable, diagnostics


def build_judge_messages(lanes: list[dict[str, Any]]) -> list[dict[str, str]]:
    reviewable, _diagnostics = partition_reviewable_lanes(lanes)
    payload = json.dumps({"lanes": _annotated_lanes(reviewable)}, indent=2)
    return [
        {
            "role": "system",
            "content": (
                "You are a UNION-MERGE for already-structured pull-request review "
                "findings from independent model lanes — NOT a filter and NOT a "
                "second reviewer. Every input finding carries an `id`. Your output "
                "must ACCOUNT FOR EVERY input id exactly once: each output issue "
                "lists the input ids it covers in `sources`, and the union of all "
                "`sources` must equal the set of input ids. Merge two findings "
                "into one issue ONLY when they describe the same defect at the "
                "same location; when unsure whether two findings are the same "
                "defect, keep them as separate issues. Never drop a finding for "
                "importance, severity, redundancy of theme, style, or quality — "
                "recall was already decided by the lanes. Do not invent issues "
                "(no output issue may have empty or unknown sources). For a "
                "merged duplicate keep the strongest severity, prefer the "
                "clearest body, and list every lane model that reported it in "
                "`models`.\n\n"
                'Return JSON only: {"issues": [{"title", "body", "severity", '
                '"file", "line", "models", "sources"}]}. severity is bug, '
                "risk, or nit. file/line may be null. Do not think at length; "
                "this is clerical merge."
            ),
        },
        {
            "role": "user",
            "content": (
                "Union-merge these lane results. Every input finding id must "
                "appear in exactly one output issue's `sources`; combine only "
                "true same-defect duplicates and keep everything else. "
                "Attribution models must be slugs from the input lanes.\n\n"
                f"{payload}"
            ),
        },
    ]


def parse_judge_issues(payload: object, *, allowed_models: list[str]) -> list[MergedIssue]:
    """Compatibility wrapper: parsed issues without source accounting."""
    issues, _sources, _bad = _parse_with_sources(payload, allowed_models=allowed_models)
    return issues


def _parse_with_sources(
    payload: object, *, allowed_models: list[str]
) -> tuple[list[MergedIssue], list[list[str]], bool]:
    """(issues, per-issue sources, sources_invalid). Structural mismatches
    raise SchemaError; per-issue source problems only set sources_invalid so
    the caller can fall back deterministically instead of failing the job."""
    if isinstance(payload, str):
        try:
            payload = extract_json_object(payload)
        except Exception as exc:  # noqa: BLE001 — treat as schema failure
            raise SchemaError(f"judge output is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SchemaError("judge output must be a JSON object")
    raw_issues = payload.get("issues")
    if raw_issues is None:
        raise SchemaError("judge output is missing an issues array")
    if not isinstance(raw_issues, list):
        raise SchemaError("judge issues must be an array")
    if len(raw_issues) > MAX_FINDINGS:
        raise SchemaError(f"judge returned more than {MAX_FINDINGS} issues")
    allowed = set(allowed_models)
    issues: list[MergedIssue] = []
    sources: list[list[str]] = []
    sources_invalid = False
    for item in raw_issues:
        issues.append(_parse_one_issue(item, allowed=allowed))
        raw_sources = item.get("sources") if isinstance(item, dict) else None
        if (
            isinstance(raw_sources, list)
            and raw_sources
            and all(isinstance(s, str) and s.strip() for s in raw_sources)
        ):
            sources.append([s.strip() for s in raw_sources])
        else:
            sources.append([])
            sources_invalid = True
    return issues, sources, sources_invalid


def _parse_one_issue(raw: object, *, allowed: set[str]) -> MergedIssue:
    if not isinstance(raw, dict):
        raise SchemaError("each judge issue must be an object")
    title = raw.get("title")
    body = raw.get("body")
    severity = raw.get("severity")
    models = raw.get("models")
    if not isinstance(title, str) or not title.strip():
        raise SchemaError("judge issue title is required")
    if not isinstance(body, str) or not body.strip():
        raise SchemaError("judge issue body is required")
    if not isinstance(severity, str) or severity.strip().lower() not in SEVERITIES:
        raise SchemaError(f"judge issue severity must be one of {', '.join(SEVERITIES)}")
    if (
        not isinstance(models, list)
        or not models
        or not all(isinstance(m, str) and m.strip() for m in models)
    ):
        raise SchemaError("judge issue models must be a non-empty array of slugs")
    cleaned_models: list[str] = []
    for model in models:
        slug = model.strip()
        if allowed and slug not in allowed:
            raise SchemaError(f"judge attributed an issue to unknown lane model {slug!r}")
        if slug not in cleaned_models:
            cleaned_models.append(slug)
    file_value = raw.get("file")
    if file_value is not None and not isinstance(file_value, str):
        raise SchemaError("judge issue file must be a string or null")
    path = file_value.strip() if isinstance(file_value, str) and file_value.strip() else None
    if path and not valid_review_path(path):
        path = None
    line = raw.get("line")
    if line is None or line == "":
        line_n = None
    elif isinstance(line, bool) or not isinstance(line, int) or line <= 0:
        raise SchemaError("judge issue line must be a positive integer or null")
    else:
        line_n = line
    return MergedIssue(
        title=title.strip()[:MAX_TITLE],
        body=body.strip()[:MAX_BODY],
        severity=severity.strip().lower(),
        file=path,
        line=line_n,
        models=cleaned_models,
    )


def _issue_from_finding(finding: dict, lane_model: str) -> MergedIssue | None:
    title = str(finding.get("title") or "").strip()
    if not title:
        return None
    severity = str(finding.get("severity") or "nit").strip().lower()
    if severity not in SEVERITIES:
        severity = "nit"
    file_value = finding.get("file")
    path = file_value.strip() if isinstance(file_value, str) and file_value.strip() else None
    if path and not valid_review_path(path):
        path = None
    line = finding.get("line")
    line_n = line if isinstance(line, int) and not isinstance(line, bool) and line > 0 else None
    if is_environmental_diagnostic(
        title=title,
        body=str(finding.get("body") or ""),
        line=line_n,
    ):
        return None
    return MergedIssue(
        title=title[:MAX_TITLE],
        body=str(finding.get("body") or "").strip()[:MAX_BODY],
        severity=severity,
        file=path,
        line=line_n,
        models=[lane_model] if lane_model else [],
    )


def _severity_sorted(issues: list[MergedIssue]) -> list[MergedIssue]:
    return sorted(issues, key=lambda i: -_SEVERITY_RANK.get(i.severity, 0))


def _capped(issues: list[MergedIssue], context: str) -> list[MergedIssue]:
    """Severity-sorted publishing cap — loud, never silent: the strongest
    findings are kept and the truncation is logged."""
    ordered = _severity_sorted(issues)
    if len(ordered) > MAX_FINDINGS:
        print(
            f"judge {context}: {len(ordered)} findings exceed the publishing "
            f"cap ({MAX_FINDINGS}); keeping the strongest severities and "
            f"dropping {len(ordered) - MAX_FINDINGS}"
        )
    return ordered[:MAX_FINDINGS]


def deterministic_union(lanes: list[dict[str, Any]]) -> list[MergedIssue]:
    """Recall-safe fallback merge: concatenate every lane's findings, merging
    only duplicates whose location, title, and evidence agree. A merged
    duplicate keeps the STRONGEST severity and the longer body, matching the
    LLM contract. Noisier than a good LLM merge, but it cannot lose distinct
    same-title findings; if the result exceeds the publishing cap, the
    strongest severities are kept."""
    union: list[MergedIssue] = []
    for lane in lanes:
        lane_model = str(lane.get("model") or "")
        for finding in lane.get("findings") or []:
            if not isinstance(finding, dict):
                continue
            issue = _issue_from_finding(finding, lane_model)
            if issue is None:
                continue
            union.append(issue)
    merged, _absorbed = deduplicate_issues(union)
    return _capped(merged, "union")


def _deduplicate_judge_rows(
    issues: list[MergedIssue], sources: list[list[str]]
) -> tuple[list[MergedIssue], list[list[str]], int]:
    """Keep issue/source rows aligned while absorbing judge duplicates."""
    kept_issues: list[MergedIssue] = []
    kept_sources: list[list[str]] = []
    absorbed = 0
    for issue, issue_sources in zip(issues, sources, strict=True):
        unique_sources = list(dict.fromkeys(issue_sources))
        for index, existing in enumerate(kept_issues):
            if same_merged_issue(existing, issue):
                absorb_merged_issue(existing, issue)
                for source in unique_sources:
                    if source not in kept_sources[index]:
                        kept_sources[index].append(source)
                absorbed += 1
                break
        else:
            kept_issues.append(issue)
            kept_sources.append(unique_sources)
    return kept_issues, kept_sources, absorbed


def _verify_coverage(
    issues: list[MergedIssue],
    sources: list[list[str]],
    sources_invalid: bool,
    lanes: list[dict[str, Any]],
) -> tuple[list[MergedIssue], str]:
    """Identity-based recall verification.

    Every input finding id must be accounted for by some output issue's
    sources. Missing ids are REPAIRED by appending those findings verbatim
    (the judge's merge work is kept). Untrustworthy accounting — invalid or
    unknown source ids — replaces the judge output with the deterministic
    union. Returns (issues, mode) where mode is 'merged', 'repaired(+N)',
    or 'union-fallback'."""
    input_ids: dict[str, tuple[dict, str]] = {}
    for lane_index, lane in enumerate(lanes):
        lane_model = str(lane.get("model") or "")
        for finding_index, finding in enumerate(lane.get("findings") or []):
            if isinstance(finding, dict):
                input_ids[_finding_id(lane_index, finding_index)] = (finding, lane_model)

    if sources_invalid:
        print(
            "judge coverage: one or more issues had missing/empty sources; "
            "using the deterministic union"
        )
        return deterministic_union(lanes), "union-fallback"

    issues, sources, judge_duplicates = _deduplicate_judge_rows(issues, sources)

    claim_counts: dict[str, int] = {}
    for issue_sources in sources:
        for source in issue_sources:
            claim_counts[source] = claim_counts.get(source, 0) + 1
    repeated = sorted(source for source, count in claim_counts.items() if count > 1)
    if repeated:
        print(
            f"judge coverage: source id(s) {repeated[:5]} were used by multiple "
            "distinct issues; using the deterministic union"
        )
        return deterministic_union(lanes), "union-fallback"

    claimed = {s for issue_sources in sources for s in issue_sources}
    unknown = claimed - set(input_ids)
    if unknown:
        print(
            f"judge coverage: unknown source id(s) {sorted(unknown)[:5]} — "
            "possible fabrication; using the deterministic union"
        )
        return deterministic_union(lanes), "union-fallback"

    # Merge LEGALITY: the contract is same-defect-same-LOCATION, so an issue
    # whose sources span different files or distant lines cannot be a true
    # duplicate merge — the model is compressing distinct findings while
    # keeping its accounting "legal". Split such issues back into their
    # constituent findings verbatim.
    kept: list[MergedIssue] = []
    split = 0
    for issue, issue_sources in zip(issues, sources, strict=True):
        if _legal_merge(issue_sources, input_ids):
            kept.append(issue)
            continue
        for fid in issue_sources:
            finding, lane_model = input_ids[fid]
            restored_issue = _issue_from_finding(finding, lane_model)
            if restored_issue is not None:
                kept.append(restored_issue)
                split += 1

    missing = [fid for fid in input_ids if fid not in claimed]
    restored = 0
    for fid in missing:
        finding, lane_model = input_ids[fid]
        issue = _issue_from_finding(finding, lane_model)
        if issue is not None:
            kept.append(issue)
            restored += 1
    kept, repair_duplicates = deduplicate_issues(kept)
    duplicate_count = judge_duplicates + repair_duplicates
    if not split and not restored and not duplicate_count:
        return kept, "merged"
    parts = []
    if split:
        parts.append(f"split {split} finding(s) out of over-broad merges")
    if restored:
        parts.append(f"restored {restored} unaccounted finding(s)")
    if duplicate_count:
        parts.append(f"suppressed {duplicate_count} duplicate issue(s)")
    print("judge coverage: " + "; ".join(parts))
    mode = "repaired"
    if split:
        mode += f"(split+{split})"
    if restored:
        mode += f"(+{restored})"
    if duplicate_count:
        mode += f"(deduped+{duplicate_count})"
    return _capped(kept, "repair"), mode


_MERGE_LINE_TOLERANCE = 5


def _legal_merge(issue_sources: list[str], input_ids: dict[str, tuple[dict, str]]) -> bool:
    """True when the merged sources plausibly describe ONE defect at ONE
    location: every source shares the same file, and their line anchors sit
    within a small window (all-null lines are allowed; mixing anchored and
    unanchored findings is not a same-location merge)."""
    if len(issue_sources) <= 1:
        return True
    files = set()
    lines: list[int | None] = []
    for fid in issue_sources:
        finding, _model = input_ids[fid]
        file_value = finding.get("file")
        files.add(file_value if isinstance(file_value, str) else None)
        line = finding.get("line")
        lines.append(line if isinstance(line, int) and not isinstance(line, bool) else None)
    if len(files) > 1:
        return False
    anchored = [line for line in lines if line is not None]
    if not anchored:
        return True
    if len(anchored) != len(lines):
        return False
    return max(anchored) - min(anchored) <= _MERGE_LINE_TOLERANCE


def run_llm_judge(
    *,
    model: str,
    lanes: list[dict[str, Any]],
    api_key: str,
    timeout: int = 180,
    chat: ChatFn | None = None,
    provider_data_collection: str | None = None,
    provider_zdr: bool = False,
) -> tuple[list[MergedIssue], str, float | None]:
    """Returns (issues, mode, cost).

    ``mode`` is ``merged`` when the judge accounted for every input finding,
    ``repaired(+N)`` when N unaccounted findings were restored verbatim,
    ``union-fallback`` when the judge's accounting could not be trusted, or
    ``skipped-diagnostics`` when no code finding remained after temporary
    review-environment failures were separated. ``cost`` is the judge request's
    OpenRouter credit cost (USD), or None when unreported/not called.
    """
    lanes, diagnostics = partition_reviewable_lanes(lanes)
    for lane_model, title in diagnostics:
        print(
            f"lane diagnostic from {lane_model or 'unknown'}: {title} "
            "(not published as a code finding)"
        )
    if diagnostics and not any(lane.get("findings") for lane in lanes):
        return [], "skipped-diagnostics", None

    allowed = [str(lane.get("model")) for lane in lanes if isinstance(lane.get("model"), str)]
    send = chat or (lambda payload: openrouter_chat(api_key, payload, timeout=timeout))
    payload: dict[str, Any] = {
        "model": model,
        "messages": build_judge_messages(lanes),
        "response_format": {"type": "json_schema", "json_schema": judge_json_schema()},
        "reasoning": dict(JUDGE_REASONING),
        "usage": {"include": True},
    }
    if _is_gemini_model(model):
        payload["max_tokens"] = MAX_GEMINI_RESPONSE_TOKENS
    if provider_data_collection not in {None, "allow", "deny"}:
        raise ActionError("provider_data_collection must be allow, deny, or unset")
    if provider_data_collection or provider_zdr:
        provider_policy: dict[str, Any] = {}
        if provider_data_collection:
            provider_policy["data_collection"] = provider_data_collection
        if provider_zdr:
            provider_policy["zdr"] = True
        payload["provider"] = provider_policy
    try:
        response = send(payload)
    except Exception as exc:  # noqa: BLE001
        raise ActionError(f"judge OpenRouter call failed: {redact(str(exc))}") from exc
    try:
        content = response_message_text(response)
    except Exception as exc:  # noqa: BLE001
        raise SchemaError(f"judge response shape is invalid: {exc}") from exc
    if not content.strip():
        raise SchemaError("judge returned an empty assistant message")
    issues, sources, sources_invalid = _parse_with_sources(content, allowed_models=allowed)
    merged, mode = _verify_coverage(issues, sources, sources_invalid, lanes)
    return merged, mode, _response_cost(response)


def _response_cost(response: dict[str, Any]) -> float | None:
    block = response.get("usage")
    if not isinstance(block, dict):
        return None
    return _response_spend(block)
