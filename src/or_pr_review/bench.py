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
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from or_pr_review.collect import CollectedReview, DiffPlan, truncate_diff
from or_pr_review.errors import ActionError
from or_pr_review.harness import run_lane
from or_pr_review.prompt import build_messages, changed_paths_from_diff
from or_pr_review.schema import MAX_COVERAGE_ENTRIES, SEVERITIES


LABEL_CONTEXTS = ("diff", "file", "repo")
ADJUDICATION_VERDICTS = ("false_positive", "true_positive_unlabeled")


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
            l
            for l in labels
            if (severity is None or l.severity == severity)
            and (context is None or l.context == context)
        ]
        hit = sum(1 for l in pool if self.matched.get(l.id))
        return hit, len(pool)

    def severity_agreement(self, labels: tuple[Label, ...]) -> tuple[int, int]:
        """Of the labels matched, how many were hit at the label's own severity."""
        hit_labels = [l for l in labels if self.matched.get(l.id)]
        agree = sum(1 for l in hit_labels if l.severity in self.matched[l.id])
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


def collected_from_fixture(fixture: Fixture) -> CollectedReview:
    truncation = truncate_diff(fixture.diff, fixture.max_diff_kb or 1024)
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
    import os

    if args.runs < 1:
        raise ActionError("--runs must be at least 1")
    fixture = load_fixture(Path(args.fixture))
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ActionError("OPENROUTER_API_KEY is not set; `bench run` spends real tokens")
    collected = collected_from_fixture(fixture)
    messages = build_messages(
        collected, custom_instructions=fixture.custom_instructions
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
    if stale:
        # Leftovers from an earlier batch would be globbed by `score` and
        # silently mix experiments.
        print(f"clearing {len(stale)} stale run file(s) from {out_dir}")
        for path in stale:
            path.unlink()
    failures = 0
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
            provider_order=(
                [p.strip() for p in args.provider.split(",") if p.strip()]
                if args.provider
                else None
            ),
            expect_coverage=True,
            expected_paths=expected_paths,
        )
        payload = lane.to_dict()
        out_path = out_dir / f"run-{index}.json"
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        status = "ok" if lane.ok else f"FAILED: {lane.error}"
        if not lane.ok:
            failures += 1
        via = f", via {lane.provider}" if lane.provider else ""
        print(
            f"  -> {out_path} ({status}; {len(lane.findings)} finding(s), "
            f"{lane.tool_rounds or 0} tool round(s){via})"
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
        "--provider",
        default="",
        help="pin OpenRouter provider routing (comma-separated order, no fallbacks), "
        "e.g. 'baseten' — for provider bake-offs",
    )
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
