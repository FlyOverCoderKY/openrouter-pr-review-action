"""Offline recall bench: replay review fixtures through the lane locally.

A fixture is a directory that freezes one review as data — no GitHub, no
posting, no live PR required:

    fixture.json   review metadata (see load_fixture)
    diff.patch     the embedded diff, exactly as a live run would collect it
    checkout/      the inert worktree the read-only tools run against
    labels.json    golden findings the lane is scored against

labels.json is a list of label objects:

    {"id": "B1", "severity": "bug", "file": "rules.py",
     "title": "Stale 2025 cap", "keywords": ["2025", "stale cap"]}

A reported finding matches a label when it cites the label's file (or the
label names no file) and any keyword regex matches its title+body,
case-insensitively. Keywords should be distinctive fragments a correct
finding could not avoid mentioning.

Usage (needs OPENROUTER_API_KEY in the environment for `run`):

    python -m or_pr_review.bench run bench/fixtures/planted-mini \
        --model x-ai/grok-4.6 --runs 3 --out /tmp/bench-out
    python -m or_pr_review.bench score bench/fixtures/planted-mini /tmp/bench-out

`score` is deterministic and offline; `run` spends real tokens. Real-PR
fixtures captured from private repositories must live OUTSIDE the repo
(e.g. bench/fixtures-local/, which is gitignored).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from or_pr_review.collect import CollectedReview, DiffPlan, Truncation
from or_pr_review.errors import ActionError
from or_pr_review.harness import run_lane
from or_pr_review.prompt import build_messages, changed_paths_from_diff
from or_pr_review.schema import SEVERITIES


@dataclass(frozen=True)
class Label:
    id: str
    severity: str
    file: str | None
    title: str
    keywords: tuple[str, ...]


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
    checkout: Path
    labels: tuple[Label, ...]


@dataclass
class RunScore:
    """Deterministic score of one lane run against the fixture labels."""

    matched: dict[str, list[str]] = field(default_factory=dict)  # label id -> finding titles
    unmatched_findings: list[dict] = field(default_factory=list)
    finding_count: int = 0

    def recall(self, labels: tuple[Label, ...], severity: str | None = None) -> tuple[int, int]:
        pool = [l for l in labels if severity is None or l.severity == severity]
        hit = sum(1 for l in pool if self.matched.get(l.id))
        return hit, len(pool)

    def precision(self) -> tuple[int, int]:
        matched_titles = {t for titles in self.matched.values() for t in titles}
        return len(matched_titles), self.finding_count


def load_fixture(fixture_dir: Path) -> Fixture:
    meta_path = fixture_dir / "fixture.json"
    if not meta_path.is_file():
        raise ActionError(f"{meta_path} does not exist")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    diff_path = fixture_dir / meta.get("diff_file", "diff.patch")
    checkout = fixture_dir / meta.get("checkout_dir", "checkout")
    labels_path = fixture_dir / meta.get("labels_file", "labels.json")
    if not checkout.is_dir():
        raise ActionError(f"fixture checkout dir {checkout} does not exist")
    labels = tuple(_parse_label(item) for item in json.loads(labels_path.read_text(encoding="utf-8")))
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
        checkout=checkout,
        labels=labels,
    )


def _parse_label(item: dict) -> Label:
    severity = item.get("severity", "")
    if severity not in SEVERITIES:
        raise ActionError(f"label {item.get('id')!r} severity must be one of {SEVERITIES}")
    keywords = tuple(item.get("keywords", ()))
    if not keywords:
        raise ActionError(f"label {item.get('id')!r} needs at least one keyword")
    for keyword in keywords:
        re.compile(keyword)  # fail fast on bad regexes
    return Label(
        id=str(item["id"]),
        severity=severity,
        file=item.get("file"),
        title=item.get("title", ""),
        keywords=keywords,
    )


def collected_from_fixture(fixture: Fixture) -> CollectedReview:
    size = len(fixture.diff.encode("utf-8"))
    return CollectedReview(
        pr_number=fixture.pr_number,
        title=fixture.title,
        body=fixture.body,
        head_sha=fixture.head_sha,
        base_ref=fixture.base_ref,
        head_ref=fixture.head_ref,
        plan=DiffPlan("full-pr", "full-pr", None, fixture.head_sha, None),
        truncation=Truncation(fixture.diff, False, size, size, 1024),
        mode="initial",
    )


def match_finding(finding: dict, label: Label) -> bool:
    if label.file is not None:
        file_value = finding.get("file") or ""
        if not (file_value == label.file or file_value.endswith("/" + label.file)):
            return False
    haystack = f"{finding.get('title', '')}\n{finding.get('body', '')}"
    return any(re.search(keyword, haystack, re.IGNORECASE) for keyword in label.keywords)


def score_run(findings: list[dict], labels: tuple[Label, ...]) -> RunScore:
    score = RunScore(finding_count=len(findings))
    for finding in findings:
        hit_any = False
        for label in labels:
            if match_finding(finding, label):
                score.matched.setdefault(label.id, []).append(finding.get("title", ""))
                hit_any = True
        if not hit_any:
            score.unmatched_findings.append(finding)
    return score


def _cmd_run(args: argparse.Namespace) -> int:
    import os

    fixture = load_fixture(Path(args.fixture))
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ActionError("OPENROUTER_API_KEY is not set; `bench run` spends real tokens")
    collected = collected_from_fixture(fixture)
    messages = build_messages(
        collected, custom_instructions=fixture.custom_instructions
    )
    expected_paths = set(changed_paths_from_diff(fixture.diff))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(args.runs):
        print(f"bench run {index + 1}/{args.runs}: model={args.model}")
        lane = run_lane(
            model=args.model,
            messages=[dict(message) for message in messages],
            api_key=api_key,
            workspace=fixture.checkout,
            max_tool_turns=args.max_tool_turns,
            effort=args.effort,
            timeout=args.timeout,
            expect_coverage=True,
            expected_paths=expected_paths,
        )
        payload = lane.to_dict()
        out_path = out_dir / f"run-{index}.json"
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        status = "ok" if lane.ok else f"FAILED: {lane.error}"
        print(
            f"  -> {out_path} ({status}; {len(lane.findings)} finding(s), "
            f"{lane.tool_rounds or 0} tool round(s))"
        )
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    fixture = load_fixture(Path(args.fixture))
    run_files = sorted(Path(args.out).glob("run-*.json"))
    if not run_files:
        raise ActionError(f"no run-*.json files in {args.out}")
    per_run: list[RunScore] = []
    for run_file in run_files:
        payload = json.loads(run_file.read_text(encoding="utf-8"))
        if not payload.get("ok"):
            print(f"{run_file.name}: lane failed ({payload.get('error')}); scored as zero")
            per_run.append(RunScore())
            continue
        per_run.append(score_run(payload.get("findings", []), fixture.labels))

    print(f"\nfixture `{fixture.name}`: {len(fixture.labels)} label(s), {len(per_run)} run(s)\n")
    header = ["run", "recall", *[f"recall:{s}" for s in SEVERITIES], "precision", "findings"]
    print(" | ".join(header))
    for index, score in enumerate(per_run):
        hit, total = score.recall(fixture.labels)
        cells = [f"run-{index}", f"{hit}/{total}"]
        for severity in SEVERITIES:
            s_hit, s_total = score.recall(fixture.labels, severity)
            cells.append(f"{s_hit}/{s_total}")
        p_hit, p_total = score.precision()
        cells.append(f"{p_hit}/{p_total}")
        cells.append(str(score.finding_count))
        print(" | ".join(cells))

    missed = [
        label
        for label in fixture.labels
        if not any(score.matched.get(label.id) for score in per_run)
    ]
    if missed:
        print("\nlabels missed by every run:")
        for label in missed:
            print(f"  - {label.id} [{label.severity}] {label.file or '(no file)'} — {label.title}")
    for index, score in enumerate(per_run):
        if score.unmatched_findings:
            print(f"\nrun-{index} findings matching no label (triage: new true positive or noise?):")
            for finding in score.unmatched_findings:
                print(
                    f"  - [{finding.get('severity')}] {finding.get('file') or '(no file)'} — "
                    f"{finding.get('title')}"
                )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="or-pr-review-bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="replay a fixture through the lane (spends tokens)")
    run_parser.add_argument("fixture")
    run_parser.add_argument("--model", default="x-ai/grok-4.6")
    run_parser.add_argument("--runs", type=int, default=1)
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--max-tool-turns", type=int, default=50)
    run_parser.add_argument("--effort", default="high")
    run_parser.add_argument("--timeout", type=int, default=180)
    run_parser.set_defaults(func=_cmd_run)

    score_parser = sub.add_parser("score", help="score saved runs against fixture labels (offline)")
    score_parser.add_argument("fixture")
    score_parser.add_argument("out")
    score_parser.set_defaults(func=_cmd_score)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ActionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
