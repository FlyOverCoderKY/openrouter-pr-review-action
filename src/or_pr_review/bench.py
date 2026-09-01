"""Offline recall bench: replay review fixtures through the lane locally.

A fixture is a directory that freezes one review as data — no GitHub, no
posting, no live PR required:

    fixture.json         review metadata (see load_fixture)
    diff.patch           the full collected diff; the optional fixture.json
                         field `max_diff_kb` applies the production embed cap
    checkout/            the inert worktree the read-only tools run against
    labels.json          golden findings the lane is scored against
    adjudications.json   optional curated verdicts for recurring unmatched
                         findings (true_positive_unlabeled | false_positive)

labels.json is a list of label objects; `context` (required) records the
minimum context needed to find the plant — diff | file | repo — and score
reports recall per stratum:

    {"id": "B1", "severity": "bug", "file": "rules.py", "context": "diff",
     "title": "Stale 2027 cap", "keywords": ["2025", "stale cap"]}

A reported finding matches a label when it cites the label's file (or the
label names no file) and any keyword regex matches its title+body,
case-insensitively. Keywords should be distinctive fragments a correct
finding could not avoid mentioning. Recall is a DETECTION metric: a finding
matches regardless of the severity it reported; the score table's `sev-agree`
column separately reports how many matched labels were hit at the label's own
severity. Findings matching no label are three-way classified via the
adjudications (never auto-false); the `noise` column (adjudicated-false plus
unadjudicated) is the oversensitivity headline, and on a zero-label clean
twin fixture it IS the score.

Usage (needs OPENROUTER_API_KEY in the environment for `run`):

    python -m or_pr_review.bench run bench/fixtures/planted-mini \
        --model x-ai/grok-4.6 --out /tmp/bench-out
    python -m or_pr_review.bench score bench/fixtures/planted-mini /tmp/bench-out

`score` is deterministic and offline; `run` spends real tokens. Real-PR
fixtures captured from private repositories must live OUTSIDE the repo
(e.g. bench/fixtures-local/, which is gitignored).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from or_pr_review import __version__
from or_pr_review.collect import CollectedReview, DiffPlan, pack_diff, truncate_diff
from or_pr_review.errors import ActionError
from or_pr_review.harness import DEFAULT_LANE_TIMEOUT_SECONDS, openrouter_chat, run_lane
from or_pr_review.judge import (
    JUDGE_REASONING,
    build_judge_messages,
    judge_json_schema,
    run_llm_judge,
)
from or_pr_review.models import DEFAULT_JUDGE_MODEL, parse_slug
from or_pr_review.prompt import build_messages, changed_paths_from_diff, parse_path_profiles
from or_pr_review.redaction import redact
from or_pr_review.schema import MAX_COVERAGE_ENTRIES, SEVERITIES

LABEL_CONTEXTS = ("diff", "file", "repo")
ADJUDICATION_VERDICTS = ("false_positive", "true_positive_unlabeled")
JUDGE_FIXTURE_SCHEMA_VERSION = 1
JUDGE_RUN_SCHEMA_VERSION = 1
JUDGE_VERDICTS = ("clean", "issues")
JUDGE_LINE_TOLERANCE = 5
JUDGE_MAX_RUNS = 20


@dataclass(frozen=True)
class Label:
    id: str
    severity: str
    file: str | None
    title: str
    keywords: tuple[str, ...]
    # Minimum context needed to find this plant: visible in the embedded
    # diff, requires reading the changed file beyond the hunks, or requires
    # tool use on files outside the diff. Changes must not regress recall in
    # any stratum — aggregate recall can hide local-context loss.
    context: str = "diff"


@dataclass(frozen=True)
class Adjudication:
    """A curated verdict for a recurring unmatched finding.

    Unmatched findings are three-way classified — adjudicated true positive,
    adjudicated false positive, or unadjudicated — rather than all counting
    against precision. Verdicts live in the fixture's adjudications.json.
    """

    id: str
    verdict: str  # false_positive | true_positive_unlabeled
    file: str | None
    keywords: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class Fixture:
    name: str
    title: str
    body: str
    pr_number: int
    head_sha: str
    base_ref: str
    head_ref: str
    custom_instructions: str
    diff: str
    max_diff_kb: int | None
    checkout: Path
    labels: tuple[Label, ...]
    adjudications: tuple[Adjudication, ...] = ()
    path_profiles: list[dict] | None = None


@dataclass
class RunScore:
    """Deterministic score of one lane run against the fixture labels."""

    # label id -> severities of the findings that matched it
    matched: dict[str, list[str]] = field(default_factory=dict)
    # Findings matching no label, three-way classified via adjudications.
    # Adjudicated entries are (finding, adjudication_id) so the listing can
    # show which rule fired.
    adjudicated_tp: list[tuple[dict, str]] = field(default_factory=list)
    adjudicated_fp: list[tuple[dict, str]] = field(default_factory=list)
    unadjudicated: list[dict] = field(default_factory=list)
    finding_count: int = 0

    def recall(
        self,
        labels: tuple[Label, ...],
        severity: str | None = None,
        context: str | None = None,
    ) -> tuple[int, int]:
        pool = [
            label
            for label in labels
            if (severity is None or label.severity == severity)
            and (context is None or label.context == context)
        ]
        hit = sum(1 for label in pool if self.matched.get(label.id))
        return hit, len(pool)

    def severity_agreement(self, labels: tuple[Label, ...]) -> tuple[int, int]:
        """Of the labels matched, how many were hit at the label's own severity."""
        hit_labels = [label for label in labels if self.matched.get(label.id)]
        agree = sum(
            1 for label in hit_labels if label.severity in self.matched[label.id]
        )
        return agree, len(hit_labels)

    def precision(self) -> tuple[int, int]:
        """(true positives, adjudicated total) — unadjudicated findings are a
        third class reported separately, never auto-counted as false."""
        label_matched = self.finding_count - (
            len(self.adjudicated_tp) + len(self.adjudicated_fp) + len(self.unadjudicated)
        )
        true_positive = label_matched + len(self.adjudicated_tp)
        return true_positive, true_positive + len(self.adjudicated_fp)

    def noise(self) -> tuple[int, int]:
        """(adjudicated-false + unadjudicated, total findings).

        The headline oversensitivity metric: padding moves it even when the
        adjudicated precision column stays perfect, and on a clean twin it IS
        the score."""
        return len(self.adjudicated_fp) + len(self.unadjudicated), self.finding_count


@dataclass(frozen=True)
class JudgeExpectation:
    """One unique issue the merge judge should emit for a synthetic set."""

    id: str
    title: str
    file: str | None
    line: int | None
    keywords: tuple[str, ...]
    recall_critical: bool = False


@dataclass(frozen=True)
class JudgeFixture:
    """Synthetic A/B or A/B/C lane set with deterministic ground truth."""

    name: str
    description: str
    expected_verdict: str
    lanes: tuple[dict[str, Any], ...]
    expected_issues: tuple[JudgeExpectation, ...]
    source: Path
    sha256: str


@dataclass(frozen=True)
class JudgeScore:
    """Offline score for one saved judge result."""

    expected_count: int
    output_count: int
    true_positive_count: int
    duplicate_count: int
    missed_ids: tuple[str, ...]
    unmatched_titles: tuple[str, ...]
    critical_hit: int
    critical_total: int
    expected_verdict: str
    actual_verdict: str

    @property
    def precision(self) -> float:
        return self.true_positive_count / self.output_count if self.output_count else 1.0

    @property
    def recall(self) -> float:
        return self.true_positive_count / self.expected_count if self.expected_count else 1.0

    @property
    def duplicate_rate(self) -> float:
        return self.duplicate_count / self.output_count if self.output_count else 0.0

    @property
    def verdict_correct(self) -> bool:
        return self.expected_verdict == self.actual_verdict


def _confined(fixture_dir: Path, relative: str, kind: str) -> Path:
    """Resolve a fixture-relative path and refuse escapes from the fixture dir."""
    base = fixture_dir.resolve()
    resolved = (base / relative).resolve()
    if resolved != base and base not in resolved.parents:
        raise ActionError(f"fixture {kind} {relative!r} escapes the fixture directory")
    return resolved


def load_fixture(fixture_dir: Path) -> Fixture:
    meta_path = fixture_dir / "fixture.json"
    if not meta_path.is_file():
        raise ActionError(f"{meta_path} does not exist")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ActionError(f"{meta_path} is not valid JSON: {exc}") from exc
    diff_path = _confined(fixture_dir, meta.get("diff_file", "diff.patch"), "diff_file")
    checkout = _confined(fixture_dir, meta.get("checkout_dir", "checkout"), "checkout_dir")
    labels_path = _confined(fixture_dir, meta.get("labels_file", "labels.json"), "labels_file")
    if not checkout.is_dir():
        raise ActionError(f"fixture checkout dir {checkout} does not exist")
    if not diff_path.is_file():
        raise ActionError(f"fixture diff {diff_path} does not exist")
    if not labels_path.is_file():
        raise ActionError(f"fixture labels {labels_path} does not exist")
    try:
        raw_labels = json.loads(labels_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ActionError(f"{labels_path} is not valid JSON: {exc}") from exc
    labels = tuple(_parse_label(item) for item in raw_labels)
    adjudications: tuple[Adjudication, ...] = ()
    adjudications_path = fixture_dir / "adjudications.json"
    if adjudications_path.is_file():
        try:
            raw_adjudications = json.loads(adjudications_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ActionError(f"{adjudications_path} is not valid JSON: {exc}") from exc
        adjudications = tuple(_parse_adjudication(item) for item in raw_adjudications)
        ids = [adjudication.id for adjudication in adjudications]
        if len(ids) != len(set(ids)):
            raise ActionError(f"{adjudications_path} has duplicate adjudication ids")
    max_diff_kb = meta.get("max_diff_kb")
    if max_diff_kb is not None and (
        isinstance(max_diff_kb, bool) or not isinstance(max_diff_kb, int) or max_diff_kb <= 0
    ):
        raise ActionError("fixture max_diff_kb must be a positive integer when present")
    return Fixture(
        name=fixture_dir.name,
        title=meta.get("title", ""),
        body=meta.get("body", ""),
        pr_number=int(meta.get("pr_number", 1)),
        head_sha=meta.get("head_sha", "f" * 40),
        base_ref=meta.get("base_ref", "main"),
        head_ref=meta.get("head_ref", "feature"),
        custom_instructions=meta.get("custom_instructions", ""),
        diff=diff_path.read_text(encoding="utf-8"),
        max_diff_kb=max_diff_kb,
        checkout=checkout,
        labels=labels,
        adjudications=adjudications,
        path_profiles=parse_path_profiles(
            json.dumps(meta["path_profiles"]) if meta.get("path_profiles") else None
        ),
    )


def _parse_label(item: object) -> Label:
    if not isinstance(item, dict):
        raise ActionError("each label must be a JSON object")
    if "id" not in item:
        raise ActionError("a label is missing its id")
    severity = item.get("severity", "")
    if severity not in SEVERITIES:
        raise ActionError(f"label {item.get('id')!r} severity must be one of {SEVERITIES}")
    raw_keywords = item.get("keywords")
    if not isinstance(raw_keywords, list) or not raw_keywords:
        # A bare string would silently explode into one regex per character.
        raise ActionError(f"label {item.get('id')!r} keywords must be a non-empty list")
    keywords: list[str] = []
    for keyword in raw_keywords:
        if not isinstance(keyword, str) or not keyword.strip():
            raise ActionError(
                f"label {item.get('id')!r} has an empty keyword, which would match "
                "every finding"
            )
        try:
            re.compile(keyword)
        except re.error as exc:
            raise ActionError(
                f"label {item.get('id')!r} keyword {keyword!r} is not a valid regex: {exc}"
            ) from exc
        keywords.append(keyword)
    context = item.get("context")
    if context not in LABEL_CONTEXTS:
        # Explicit, not defaulted: silently bucketing unannotated labels as
        # "diff" would make the strata line claim tracking that isn't there.
        raise ActionError(
            f"label {item.get('id')!r} context must be one of {LABEL_CONTEXTS}"
        )
    return Label(
        id=str(item["id"]),
        severity=severity,
        file=item.get("file"),
        title=item.get("title", ""),
        keywords=tuple(keywords),
        context=context,
    )


def _parse_adjudication(item: object) -> Adjudication:
    if not isinstance(item, dict):
        raise ActionError("each adjudication must be a JSON object")
    if "id" not in item:
        raise ActionError("an adjudication is missing its id")
    verdict = item.get("verdict", "")
    if verdict not in ADJUDICATION_VERDICTS:
        raise ActionError(
            f"adjudication {item.get('id')!r} verdict must be one of {ADJUDICATION_VERDICTS}"
        )
    raw_keywords = item.get("keywords")
    if not isinstance(raw_keywords, list) or not raw_keywords:
        raise ActionError(f"adjudication {item.get('id')!r} keywords must be a non-empty list")
    for keyword in raw_keywords:
        if not isinstance(keyword, str) or not keyword.strip():
            raise ActionError(f"adjudication {item.get('id')!r} has an empty keyword")
        try:
            re.compile(keyword)
        except re.error as exc:
            raise ActionError(
                f"adjudication {item.get('id')!r} keyword {keyword!r} is not a valid "
                f"regex: {exc}"
            ) from exc
    return Adjudication(
        id=str(item["id"]),
        verdict=verdict,
        file=item.get("file"),
        keywords=tuple(raw_keywords),
        note=item.get("note", ""),
    )


def collected_from_fixture(
    fixture: Fixture, *, legacy_truncation: bool = False
) -> CollectedReview:
    """Apply the production embed cap; `legacy_truncation` replays the
    pre-triage raw byte cut for A/B baselines."""
    max_diff_kb = fixture.max_diff_kb or 1024
    if legacy_truncation:
        truncation = truncate_diff(fixture.diff, max_diff_kb)
    else:
        gitattributes = fixture.checkout / ".gitattributes"
        truncation = pack_diff(
            fixture.diff,
            max_diff_kb,
            gitattributes_text=(
                gitattributes.read_text(encoding="utf-8", errors="replace")
                if gitattributes.is_file()
                else ""
            ),
        )
    return CollectedReview(
        pr_number=fixture.pr_number,
        title=fixture.title,
        body=fixture.body,
        head_sha=fixture.head_sha,
        base_ref=fixture.base_ref,
        head_ref=fixture.head_ref,
        plan=DiffPlan("full-pr", "full-pr", None, fixture.head_sha, None),
        truncation=truncation,
        mode="initial",
    )


def match_finding(finding: dict, label: Label | Adjudication) -> bool:
    if label.file is not None:
        file_value = finding.get("file") or ""
        if not (file_value == label.file or file_value.endswith("/" + label.file)):
            return False
    haystack = f"{finding.get('title', '')}\n{finding.get('body', '')}"
    return any(re.search(keyword, haystack, re.IGNORECASE) for keyword in label.keywords)


def score_run(
    findings: list[dict],
    labels: tuple[Label, ...],
    adjudications: tuple[Adjudication, ...] = (),
) -> RunScore:
    score = RunScore(finding_count=len(findings))
    for finding in findings:
        hit_any = False
        for label in labels:
            if match_finding(finding, label):
                score.matched.setdefault(label.id, []).append(finding.get("severity", ""))
                hit_any = True
        if hit_any:
            continue
        adjudication = next(
            (a for a in adjudications if match_finding(finding, a)), None
        )
        if adjudication is None:
            score.unadjudicated.append(finding)
        elif adjudication.verdict == "true_positive_unlabeled":
            score.adjudicated_tp.append((finding, adjudication.id))
        else:
            score.adjudicated_fp.append((finding, adjudication.id))
    return score


def _cmd_run(args: argparse.Namespace) -> int:
    if args.runs < 1:
        raise ActionError("--runs must be at least 1")
    lane_timeout = args.lane_timeout
    if lane_timeout is None:
        raw_lane_timeout = os.environ.get("OR_PR_REVIEW_BENCH_LANE_TIMEOUT_SECONDS", "")
        try:
            lane_timeout = (
                int(raw_lane_timeout)
                if raw_lane_timeout
                else DEFAULT_LANE_TIMEOUT_SECONDS
            )
        except ValueError as exc:
            raise ActionError("benchmark lane timeout must be a positive integer") from exc
    if lane_timeout < 1:
        raise ActionError("benchmark lane timeout must be a positive integer")
    fixture = load_fixture(Path(args.fixture))
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ActionError("OPENROUTER_API_KEY is not set; `bench run` spends real tokens")
    collected = collected_from_fixture(
        fixture, legacy_truncation=args.legacy_truncation
    )
    if collected.truncation.truncated:
        print(
            f"embed cap applied: {collected.truncation.embedded_bytes / 1024:.1f} KB "
            f"embedded of {collected.truncation.original_bytes / 1024:.1f} KB "
            f"({len(collected.truncation.stubbed_files)} stubbed, "
            f"{len(collected.truncation.dropped_files)} dropped"
            f"{', legacy raw cut' if args.legacy_truncation else ''})"
        )
    messages = build_messages(
        collected,
        custom_instructions=fixture.custom_instructions,
        path_profiles=fixture.path_profiles,
    )
    # Mirror production: coverage enforcement degrades when the diff names
    # more paths than the manifest may hold (see _coverage_expectations).
    expected_paths: set[str] | None = set(changed_paths_from_diff(collected.diff))
    if len(expected_paths) > MAX_COVERAGE_ENTRIES:
        print(
            f"notice: {len(expected_paths)} diff paths exceed the coverage cap "
            f"({MAX_COVERAGE_ENTRIES}); coverage completeness is not enforced"
        )
        expected_paths = None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = sorted(out_dir.glob("run-*.json"))
    stale.extend(sorted(out_dir.glob("progress-*.json")))
    stale.extend(sorted(out_dir.glob("progress-*.tmp")))
    if stale:
        # Final leftovers would be globbed by `score`; progress leftovers
        # would otherwise make a fresh batch look like a resumed one.
        print(f"clearing {len(stale)} stale run file(s) from {out_dir}")
        for path in stale:
            path.unlink()
    failures = 0
    for index in range(args.runs):
        print(f"bench run {index + 1}/{args.runs}: model={args.model}", flush=True)
        progress_path = out_dir / f"progress-{index}.json"

        def checkpoint(
            payload: dict[str, int | float | str], progress_path: Path = progress_path
        ) -> None:
            record: dict[str, Any] = {
                "schema": "or-pr-review/bench-progress/1",
                "model": args.model,
                **payload,
            }
            pending = progress_path.with_suffix(".tmp")
            pending.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            os.replace(pending, progress_path)

        lane = run_lane(
            model=args.model,
            messages=[dict(message) for message in messages],
            api_key=api_key,
            workspace=fixture.checkout,
            max_tool_turns=args.max_tool_turns,
            effort=args.effort,
            timeout=args.timeout,
            lane_timeout=lane_timeout,
            provider_order=(
                [p.strip() for p in args.provider.split(",") if p.strip()]
                if args.provider
                else None
            ),
            provider_data_collection=args.provider_data_collection or None,
            provider_zdr=args.provider_zdr,
            expect_coverage=True,
            expected_paths=expected_paths,
            progress=checkpoint,
        )
        payload = lane.to_dict()
        out_path = out_dir / f"run-{index}.json"
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        progress_path.unlink(missing_ok=True)
        status = "ok" if lane.ok else f"FAILED: {lane.error}"
        if not lane.ok:
            failures += 1
        via = f", via {lane.provider}" if lane.provider else ""
        print(
            f"  -> {out_path} ({status}; {len(lane.findings)} finding(s), "
            f"{lane.tool_rounds or 0} tool round(s){via})",
            flush=True,
        )
    if failures:
        print(f"{failures}/{args.runs} lane(s) failed")
        return 1
    return 0


def _fraction(hit: int, total: int) -> str:
    return f"{hit}/{total}"


def _cmd_score(args: argparse.Namespace) -> int:
    fixture = load_fixture(Path(args.fixture))
    run_files = sorted(Path(args.out).glob("run-*.json"))
    if not run_files:
        raise ActionError(f"no run-*.json files in {args.out}")
    scored: list[tuple[str, RunScore]] = []
    failed = 0
    for run_file in run_files:
        payload = json.loads(run_file.read_text(encoding="utf-8"))
        if not payload.get("ok"):
            print(f"{run_file.name}: lane failed ({payload.get('error')}); excluded from means")
            failed += 1
            continue
        scored.append(
            (
                run_file.stem,
                score_run(payload.get("findings", []), fixture.labels, fixture.adjudications),
            )
        )

    print(
        f"\nfixture `{fixture.name}`: {len(fixture.labels)} label(s), "
        f"{len(scored)} scored run(s), {failed} failed run(s)\n"
    )
    if not scored:
        print("no successful runs to score")
        return 1
    if not fixture.labels:
        # A clean-twin fixture: there is nothing to recall, so the noise
        # column IS the score — anything nonzero is oversensitivity.
        print("clean fixture (no labels): the noise column is the score")
    header = [
        "run",
        "recall",
        *[f"recall:{s}" for s in SEVERITIES],
        "sev-agree",
        "precision",
        "noise",
        "findings",
    ]
    print(" | ".join(header))
    sums: dict[str, list[int]] = {}

    def _cell(key: str, hit: int, total: int) -> str:
        sums.setdefault(key, [0, 0])
        sums[key][0] += hit
        sums[key][1] += total
        return _fraction(hit, total)

    for name, score in scored:
        cells = [name, _cell("recall", *score.recall(fixture.labels))]
        for severity in SEVERITIES:
            cells.append(_cell(f"recall:{severity}", *score.recall(fixture.labels, severity)))
        cells.append(_cell("sev-agree", *score.severity_agreement(fixture.labels)))
        cells.append(_cell("precision", *score.precision()))
        cells.append(_cell("noise", *score.noise()))
        cells.append(str(score.finding_count))
        print(" | ".join(cells))
    mean_cells = ["mean"]
    for key in ("recall", *[f"recall:{s}" for s in SEVERITIES], "sev-agree", "precision", "noise"):
        hit, total = sums[key]
        mean_cells.append(f"{100 * hit / total:.0f}%" if total else "-")
    mean_cells.append(f"{sum(s.finding_count for _, s in scored) / len(scored):.1f}")
    print(" | ".join(mean_cells))

    # Context strata: aggregate recall can hide local-context loss, so a
    # change must not regress any stratum.
    if fixture.labels:
        strata = []
        for context in LABEL_CONTEXTS:
            hit = sum(score.recall(fixture.labels, context=context)[0] for _, score in scored)
            total = sum(score.recall(fixture.labels, context=context)[1] for _, score in scored)
            if total:
                strata.append(f"{context} {hit}/{total} ({100 * hit / total:.0f}%)")
        print("recall by context: " + ", ".join(strata))
        frequency = ", ".join(
            f"{label.id} {sum(1 for _, s in scored if s.matched.get(label.id))}/{len(scored)}"
            for label in fixture.labels
        )
        print(f"label detection frequency: {frequency}")

    missed = [
        label
        for label in fixture.labels
        if not any(score.matched.get(label.id) for _, score in scored)
    ]
    if missed:
        print("\nlabels missed by every run:")
        for label in missed:
            print(
                f"  - {label.id} [{label.severity}/{label.context}] "
                f"{label.file or '(no file)'} — {label.title}"
            )
    for name, score in scored:
        listed = [
            *[(f"adjudicated TP:{aid}", finding) for finding, aid in score.adjudicated_tp],
            *[(f"adjudicated FP:{aid}", finding) for finding, aid in score.adjudicated_fp],
            *[
                ("UNADJUDICATED (triage: label, adjudicate, or prompt-fix)", finding)
                for finding in score.unadjudicated
            ],
        ]
        if listed:
            print(f"\n{name} findings matching no label:")
            for tag_label, finding in listed:
                print(
                    f"  - [{finding.get('severity')}] {finding.get('file') or '(no file)'} — "
                    f"{finding.get('title')}  <{tag_label}>"
                )
    return 0


def load_judge_fixture(path: Path) -> JudgeFixture:
    """Load and validate one committed synthetic judge fixture."""
    if not path.is_file():
        raise ActionError(f"judge fixture {path} does not exist")
    raw_bytes = path.read_bytes()
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActionError(f"judge fixture {path} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ActionError("judge fixture must be a JSON object")
    if payload.get("schema_version") != JUDGE_FIXTURE_SCHEMA_VERSION:
        raise ActionError(
            f"judge fixture schema_version must be {JUDGE_FIXTURE_SCHEMA_VERSION}"
        )
    name = payload.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        raise ActionError("judge fixture name must be a lowercase kebab-case string")
    description = payload.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ActionError(f"judge fixture {name!r} needs a description")
    verdict = payload.get("expected_verdict")
    if verdict not in JUDGE_VERDICTS:
        raise ActionError(
            f"judge fixture {name!r} expected_verdict must be one of {JUDGE_VERDICTS}"
        )

    raw_lanes = payload.get("lanes")
    if not isinstance(raw_lanes, list) or len(raw_lanes) not in (2, 3):
        raise ActionError(f"judge fixture {name!r} must contain exactly 2 or 3 lanes")
    lanes: list[dict[str, Any]] = []
    lane_models: set[str] = set()
    for lane_index, lane in enumerate(raw_lanes):
        if not isinstance(lane, dict):
            raise ActionError(f"judge fixture {name!r} lane {lane_index} must be an object")
        model = lane.get("model")
        if not isinstance(model, str):
            raise ActionError(f"judge fixture {name!r} lane {lane_index} needs a model")
        model = parse_slug(model, what=f"judge fixture lane {lane_index} model")
        if model in lane_models:
            raise ActionError(f"judge fixture {name!r} repeats lane model {model!r}")
        lane_models.add(model)
        findings = lane.get("findings")
        if not isinstance(findings, list):
            raise ActionError(
                f"judge fixture {name!r} lane {lane_index} findings must be an array"
            )
        for finding_index, finding in enumerate(findings):
            _validate_synthetic_finding(name, lane_index, finding_index, finding)
        lanes.append({"model": model, "findings": findings})

    raw_expected = payload.get("expected_issues")
    if not isinstance(raw_expected, list):
        raise ActionError(f"judge fixture {name!r} expected_issues must be an array")
    expected = tuple(_parse_judge_expectation(name, item) for item in raw_expected)
    ids = [item.id for item in expected]
    if len(ids) != len(set(ids)):
        raise ActionError(f"judge fixture {name!r} repeats an expected issue id")
    if verdict == "clean" and expected:
        raise ActionError(f"clean judge fixture {name!r} cannot have expected issues")
    if verdict == "issues" and not expected:
        raise ActionError(f"issues judge fixture {name!r} needs expected issues")
    return JudgeFixture(
        name=name,
        description=description.strip(),
        expected_verdict=verdict,
        lanes=tuple(lanes),
        expected_issues=expected,
        source=path.resolve(),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _validate_synthetic_finding(
    fixture_name: str, lane_index: int, finding_index: int, finding: object
) -> None:
    prefix = f"judge fixture {fixture_name!r} lane {lane_index} finding {finding_index}"
    if not isinstance(finding, dict):
        raise ActionError(f"{prefix} must be an object")
    for field_name in ("title", "body"):
        if not isinstance(finding.get(field_name), str) or not finding[field_name].strip():
            raise ActionError(f"{prefix} needs non-empty {field_name}")
    if finding.get("severity") not in SEVERITIES:
        raise ActionError(f"{prefix} severity must be one of {SEVERITIES}")
    file_value = finding.get("file")
    if file_value is not None and not isinstance(file_value, str):
        raise ActionError(f"{prefix} file must be a string or null")
    line = finding.get("line")
    if line is not None and (
        isinstance(line, bool) or not isinstance(line, int) or line <= 0
    ):
        raise ActionError(f"{prefix} line must be a positive integer or null")


def _parse_judge_expectation(fixture_name: str, item: object) -> JudgeExpectation:
    if not isinstance(item, dict):
        raise ActionError(f"judge fixture {fixture_name!r} expected issue must be an object")
    issue_id = item.get("id")
    if not isinstance(issue_id, str) or not re.fullmatch(r"[A-Z][A-Z0-9-]*", issue_id):
        raise ActionError(
            f"judge fixture {fixture_name!r} expected issue id must be uppercase and stable"
        )
    title = item.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ActionError(f"expected issue {issue_id!r} needs a title")
    file_value = item.get("file")
    if file_value is not None and not isinstance(file_value, str):
        raise ActionError(f"expected issue {issue_id!r} file must be a string or null")
    line = item.get("line")
    if line is not None and (
        isinstance(line, bool) or not isinstance(line, int) or line <= 0
    ):
        raise ActionError(f"expected issue {issue_id!r} line must be a positive integer or null")
    raw_keywords = item.get("keywords")
    if not isinstance(raw_keywords, list) or not raw_keywords:
        raise ActionError(f"expected issue {issue_id!r} keywords must be a non-empty array")
    keywords: list[str] = []
    for keyword in raw_keywords:
        if not isinstance(keyword, str) or not keyword.strip():
            raise ActionError(f"expected issue {issue_id!r} has an empty keyword")
        try:
            re.compile(keyword)
        except re.error as exc:
            raise ActionError(
                f"expected issue {issue_id!r} keyword {keyword!r} is invalid: {exc}"
            ) from exc
        keywords.append(keyword)
    recall_critical = item.get("recall_critical", False)
    if not isinstance(recall_critical, bool):
        raise ActionError(f"expected issue {issue_id!r} recall_critical must be boolean")
    return JudgeExpectation(
        id=issue_id,
        title=title.strip(),
        file=file_value,
        line=line,
        keywords=tuple(keywords),
        recall_critical=recall_critical,
    )


def _judge_issue_matches(issue: dict[str, Any], expected: JudgeExpectation) -> bool:
    if expected.file is not None:
        issue_file = issue.get("file")
        if not isinstance(issue_file, str) or not (
            issue_file == expected.file or issue_file.endswith("/" + expected.file)
        ):
            return False
    if expected.line is not None:
        issue_line = issue.get("line")
        if (
            isinstance(issue_line, bool)
            or not isinstance(issue_line, int)
            or abs(issue_line - expected.line) > JUDGE_LINE_TOLERANCE
        ):
            return False
    haystack = f"{issue.get('title', '')}\n{issue.get('body', '')}"
    return any(re.search(pattern, haystack, re.IGNORECASE) for pattern in expected.keywords)


def score_judge_output(
    issues: list[dict[str, Any]], fixture: JudgeFixture
) -> JudgeScore:
    """Score issues with a maximum one-to-one output-to-ground-truth match.

    One broad output cannot earn recall for several expected issues. Extra
    outputs that match an already-covered issue are counted as duplicates;
    other extras reduce precision as unsupported/noisy findings.
    """
    candidates = [
        [
            expected_index
            for expected_index, expected in enumerate(fixture.expected_issues)
            if _judge_issue_matches(issue, expected)
        ]
        for issue in issues
    ]
    expected_to_output: dict[int, int] = {}

    def assign(output_index: int, seen: set[int]) -> bool:
        for expected_index in candidates[output_index]:
            if expected_index in seen:
                continue
            seen.add(expected_index)
            previous = expected_to_output.get(expected_index)
            if previous is None or assign(previous, seen):
                expected_to_output[expected_index] = output_index
                return True
        return False

    # Least-ambiguous outputs first makes the deterministic matching easier
    # to understand while the augmenting path still finds max cardinality.
    for output_index in sorted(range(len(issues)), key=lambda i: (len(candidates[i]), i)):
        assign(output_index, set())
    matched_outputs = set(expected_to_output.values())
    matched_expected = set(expected_to_output)
    duplicate_count = sum(
        1
        for index, matches in enumerate(candidates)
        if index not in matched_outputs and any(match in matched_expected for match in matches)
    )
    missed_ids = tuple(
        expected.id
        for index, expected in enumerate(fixture.expected_issues)
        if index not in matched_expected
    )
    unmatched_titles = tuple(
        str(issue.get("title") or "(untitled)")
        for index, issue in enumerate(issues)
        if index not in matched_outputs and not candidates[index]
    )
    critical = [
        index
        for index, expected in enumerate(fixture.expected_issues)
        if expected.recall_critical
    ]
    actual_verdict = "issues" if issues else "clean"
    return JudgeScore(
        expected_count=len(fixture.expected_issues),
        output_count=len(issues),
        true_positive_count=len(matched_expected),
        duplicate_count=duplicate_count,
        missed_ids=missed_ids,
        unmatched_titles=unmatched_titles,
        critical_hit=sum(1 for index in critical if index in matched_expected),
        critical_total=len(critical),
        expected_verdict=fixture.expected_verdict,
        actual_verdict=actual_verdict,
    )


def _parse_judge_models(raw: str) -> list[str]:
    models = [part.strip() for part in raw.split(",") if part.strip()]
    if not models:
        raise ActionError("--models must contain at least one OpenRouter slug")
    parsed = [parse_slug(model, what="judge benchmark model") for model in models]
    if len(parsed) != len(set(parsed)):
        raise ActionError("--models contains duplicate slugs")
    if len(parsed) > 8:
        raise ActionError("--models is capped at 8 candidates per experiment")
    return parsed


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _merged_issue_dict(issue: object) -> dict[str, Any]:
    return {
        "title": str(getattr(issue, "title", "")),
        "body": str(getattr(issue, "body", "")),
        "severity": str(getattr(issue, "severity", "")),
        "file": getattr(issue, "file", None),
        "line": getattr(issue, "line", None),
        "models": list(getattr(issue, "models", [])),
    }


def _capturing_judge_chat(
    api_key: str, timeout: int, response_meta: dict[str, Any]
):
    """Create a live chat callback that retains only safe repeatability metadata."""

    def send(payload: dict[str, Any]) -> dict[str, Any]:
        response = openrouter_chat(api_key, payload, timeout=timeout)
        response_meta["response_id"] = response.get("id")
        response_meta["provider"] = response.get("provider")
        response_meta["served_model"] = response.get("model")
        usage = response.get("usage")
        if isinstance(usage, dict):
            response_meta["usage"] = {
                key: usage[key]
                for key in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "cached_tokens",
                )
                if isinstance(usage.get(key), (int, float))
                and not isinstance(usage.get(key), bool)
            }
        return response

    return send


def _judge_run_plan(
    out_dir: Path, fixture_name: str, models: list[str], runs: int
) -> list[tuple[str, int, Path]]:
    """Resolve the full paid-call matrix and reject path collisions up front."""
    plan: list[tuple[str, int, Path]] = []
    owners: dict[Path, tuple[str, int]] = {}
    for model in models:
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model)
        for run_index in range(runs):
            out_path = out_dir / f"judge-{fixture_name}-{safe_model}-run-{run_index}.json"
            previous = owners.get(out_path)
            if previous is not None:
                previous_model, previous_run = previous
                raise ActionError(
                    "judge output filename collision: "
                    f"{model!r} run {run_index} and {previous_model!r} run "
                    f"{previous_run} both map to {out_path.name!r}"
                )
            owners[out_path] = (model, run_index)
            if out_path.exists():
                raise ActionError(
                    f"{out_path} already exists; use a fresh output directory "
                    "so experiments cannot mix"
                )
            plan.append((model, run_index, out_path))
    return plan


def _cmd_judge_run(args: argparse.Namespace) -> int:
    """Explicitly opted-in live comparison; this is the only paid judge command."""
    if not args.allow_spend:
        raise ActionError(
            "judge-run is live and spends OpenRouter credits; re-run with --allow-spend"
        )
    if args.runs < 1:
        raise ActionError("--runs must be at least 1")
    if args.runs > JUDGE_MAX_RUNS:
        raise ActionError(f"--runs cannot exceed the safety cap of {JUDGE_MAX_RUNS}")
    fixture = load_judge_fixture(Path(args.fixture))
    models = _parse_judge_models(args.models)
    out_dir = Path(args.out)
    plan = _judge_run_plan(out_dir, fixture.name, models, args.runs)
    prompt_contract = {
        "messages": build_judge_messages(list(fixture.lanes)),
        "response_format": {
            "type": "json_schema",
            "json_schema": judge_json_schema(),
        },
        "reasoning": dict(JUDGE_REASONING),
    }
    prompt_sha256 = _json_sha256(prompt_contract)
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ActionError("OPENROUTER_API_KEY is not set; no requests were made")
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for model, run_index, out_path in plan:
        started_at = datetime.now(UTC).isoformat()
        started = time.monotonic()
        response_meta: dict[str, Any] = {}
        capturing_chat = _capturing_judge_chat(api_key, args.timeout, response_meta)

        print(
            f"judge live run {run_index + 1}/{args.runs}: "
            f"fixture={fixture.name}, model={model}"
        )
        try:
            issues, mode, cost = run_llm_judge(
                model=model,
                lanes=list(fixture.lanes),
                api_key=api_key,
                timeout=args.timeout,
                chat=capturing_chat,
                provider_data_collection="deny",
                provider_zdr=True,
            )
            record: dict[str, Any] = {
                "schema_version": JUDGE_RUN_SCHEMA_VERSION,
                "kind": "judge-benchmark-run",
                "ok": True,
                "fixture": fixture.name,
                "fixture_sha256": fixture.sha256,
                "prompt_sha256": prompt_sha256,
                "model": model,
                "run": run_index,
                "started_at": started_at,
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "harness_version": __version__,
                "production_default_judge": DEFAULT_JUDGE_MODEL,
                "mode": mode,
                "cost_usd": cost,
                "routing": "openrouter-zdr-data-collection-deny",
                **response_meta,
                "issues": [_merged_issue_dict(issue) for issue in issues],
            }
            print(
                f"  -> {out_path} ({len(issues)} issue(s), mode={mode}, "
                f"cost={'unreported' if cost is None else f'${cost:.6f}'})"
            )
        except ActionError as exc:
            failures += 1
            safe_error = redact(str(exc), extra=[api_key])
            record = {
                "schema_version": JUDGE_RUN_SCHEMA_VERSION,
                "kind": "judge-benchmark-run",
                "ok": False,
                "fixture": fixture.name,
                "fixture_sha256": fixture.sha256,
                "prompt_sha256": prompt_sha256,
                "model": model,
                "run": run_index,
                "started_at": started_at,
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "harness_version": __version__,
                "production_default_judge": DEFAULT_JUDGE_MODEL,
                "error": safe_error,
                "issues": [],
            }
            print(f"  -> {out_path} (FAILED: {safe_error})")
        out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 1 if failures else 0


def _judge_result_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("judge-*.json"))
    return []


def _cmd_judge_score(args: argparse.Namespace) -> int:
    fixture = load_judge_fixture(Path(args.fixture))
    result_files = _judge_result_files(Path(args.results))
    if not result_files:
        raise ActionError(f"no judge result JSON files found at {args.results}")
    rows: list[tuple[str, dict[str, Any], JudgeScore]] = []
    failed = 0
    for result_file in result_files:
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ActionError(f"{result_file} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ActionError(f"{result_file} must contain a JSON object")
        if payload.get("ok") is False:
            print(f"{result_file.name}: failed run excluded ({payload.get('error')})")
            failed += 1
            continue
        recorded_fixture_sha = payload.get("fixture_sha256")
        if recorded_fixture_sha is not None and recorded_fixture_sha != fixture.sha256:
            raise ActionError(
                f"{result_file.name} was produced from a different fixture revision"
            )
        issues = payload.get("issues")
        if not isinstance(issues, list) or not all(isinstance(issue, dict) for issue in issues):
            raise ActionError(f"{result_file.name} issues must be an array of objects")
        rows.append((result_file.name, payload, score_judge_output(issues, fixture)))
    print(
        f"\njudge fixture `{fixture.name}`: {len(fixture.expected_issues)} expected unique "
        f"issue(s), verdict={fixture.expected_verdict}; {len(rows)} scored, {failed} failed\n"
    )
    if not rows:
        return 1
    print("run | model | recall | precision | duplicates | critical | verdict | mode | ms | cost")
    for name, payload, score in rows:
        critical = (
            _fraction(score.critical_hit, score.critical_total)
            if score.critical_total
            else "-"
        )
        verdict = (
            f"yes ({score.actual_verdict})"
            if score.verdict_correct
            else f"NO ({score.actual_verdict}, want {score.expected_verdict})"
        )
        cost = payload.get("cost_usd")
        print(
            " | ".join(
                [
                    name,
                    str(payload.get("model") or "saved-output"),
                    _fraction(score.true_positive_count, score.expected_count),
                    _fraction(score.true_positive_count, score.output_count),
                    f"{score.duplicate_count}/{score.output_count}",
                    critical,
                    verdict,
                    str(payload.get("mode") or "unknown"),
                    str(payload.get("elapsed_ms") or "-"),
                    "-" if cost is None else f"${float(cost):.6f}",
                ]
            )
        )
        if score.missed_ids:
            print(f"  missed: {', '.join(score.missed_ids)}")
        if score.unmatched_titles:
            print(f"  unsupported/noise: {', '.join(score.unmatched_titles)}")
    print("\naggregate")
    print(
        f"  mean recall: {sum(s.recall for _, _, s in rows) / len(rows):.1%}; "
        f"mean precision: {sum(s.precision for _, _, s in rows) / len(rows):.1%}; "
        f"mean duplicate rate: "
        f"{sum(s.duplicate_rate for _, _, s in rows) / len(rows):.1%}; "
        f"verdict accuracy: "
        f"{sum(s.verdict_correct for _, _, s in rows)}/{len(rows)}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="or-pr-review-bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="replay a fixture through the lane (spends tokens)")
    run_parser.add_argument("fixture")
    run_parser.add_argument("--model", default="x-ai/grok-4.6")
    run_parser.add_argument(
        "--runs", type=int, default=3, help="lanes are nondeterministic; compare means of 3+"
    )
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--max-tool-turns", type=int, default=50)
    run_parser.add_argument(
        "--effort",
        default="",
        help="reasoning effort; empty matches the action's default (no effort field)",
    )
    run_parser.add_argument("--timeout", type=int, default=180)
    run_parser.add_argument(
        "--lane-timeout",
        type=int,
        default=None,
        help="outer lane deadline in seconds; benchmark runners may raise it for 50-turn reviews",
    )
    run_parser.add_argument(
        "--legacy-truncation",
        action="store_true",
        help="replay the pre-triage raw byte cut instead of diff-budget "
        "triage (the A/B baseline arm for over-budget fixtures)",
    )
    run_parser.add_argument(
        "--provider",
        default="",
        help="pin OpenRouter provider routing (comma-separated order, no fallbacks), "
        "e.g. 'baseten' — for provider bake-offs",
    )
    run_parser.add_argument(
        "--provider-data-collection",
        choices=("allow", "deny"),
        default="deny",
        help="set OpenRouter's provider.data_collection policy per request "
        "(default: deny)",
    )
    run_parser.add_argument(
        "--provider-zdr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require an OpenRouter zero-data-retention endpoint per request "
        "(default: enabled; use --no-provider-zdr only for non-private bake-offs)",
    )
    run_parser.set_defaults(func=_cmd_run)

    score_parser = sub.add_parser("score", help="score saved runs against fixture labels (offline)")
    score_parser.add_argument("fixture")
    score_parser.add_argument("out")
    score_parser.set_defaults(func=_cmd_score)

    judge_run_parser = sub.add_parser(
        "judge-run",
        help="compare merge-judge models on a synthetic set (LIVE; spends tokens)",
    )
    judge_run_parser.add_argument("fixture")
    judge_run_parser.add_argument(
        "--models",
        default=DEFAULT_JUDGE_MODEL,
        help="comma-separated exact OpenRouter slugs",
    )
    judge_run_parser.add_argument(
        "--runs", type=int, default=3, help="judge calls are nondeterministic; compare 3+"
    )
    judge_run_parser.add_argument("--out", required=True)
    judge_run_parser.add_argument("--timeout", type=int, default=180)
    judge_run_parser.add_argument(
        "--allow-spend",
        action="store_true",
        help="required acknowledgement that this command makes paid OpenRouter calls",
    )
    judge_run_parser.set_defaults(func=_cmd_judge_run)

    judge_score_parser = sub.add_parser(
        "judge-score",
        help="score saved judge output against synthetic ground truth (offline)",
    )
    judge_score_parser.add_argument("fixture")
    judge_score_parser.add_argument("results", help="one result JSON or a run directory")
    judge_score_parser.set_defaults(func=_cmd_judge_score)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ActionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
