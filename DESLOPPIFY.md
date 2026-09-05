# Desloppify Backlog

Implementation follow-up: R1, R2, R3 and R5 are addressed on `codex/review-loop-hardening`.
R4 (durable matrix review context), R6 and R7 remain open. The separate runner
already supplies the stronger benchmark scoring/provenance used for model comparisons;
R6/R7 are native action-benchmark limitations and do not require repeating paid runs.

Validation: 467 tests passed, 2 platform skips, plus Ruff and whitespace checks.
No model calls or live workflow runs. Canonical source reconstruction intentionally
keeps uncertain duplicate matches separate. The original scan below remains the
record of the starting defects; checked items describe behavior before these fixes.

Fresh audit of `465b7020abdcc17ba47644b8a1d1126aa44f15a2` on `codex/desloppify-fixes`, started 2026-09-04. Only this report is changed.

Prior report: `.claude/worktrees/desloppify-scan-355d6c/DESLOPPIFY.md`. Its final disposition records 60 original and 16 rescan items closed at this head. This audit preserves that report and its accepted policy decisions. New findings below concern behavior beyond those fixes; historical audit claims are not treated as current defects.

Original audit: **7 findings: 3 high-priority correctness issues and 4 medium-priority reliability/evaluation issues.** No cosmetic-only backlog was added. P1 means fix soon; no P0 emergency is claimed. Line references target the scanned commit.

## Critical Issues

- [x] R1 [P1] Preserve source severity and evidence through judge output
  - Where: `src/or_pr_review/judge.py:395-471` (`_verify_coverage`), `:503-528` (`_legal_merge`), `:240-284` (`_parse_one_issue`).
  - Why it matters: Source IDs are accounted for, but the accepted issue's severity, location, attribution, and text are not checked against those sources. A source bug can become a nit while mode remains `merged`; `fail_on=bugs` then sees no bug and the next round retires the nit. Distinct defects in the same five-line window can also be represented by one issue naming both source IDs without preserving both defects. Identity accounting alone does not guarantee the documented preservation of validated findings. The existing 2026-08-29 judge benchmark in `bench/RESULTS.md` also records unsupported same-location lumps and missed expected findings without repair/fallback; the invariant gap is therefore consistent with already observed model behavior.
  - Recommendation: Derive severity and attribution deterministically from sources; preserve canonical source evidence and constrain locations. For a clerical judge, prefer returning grouping IDs and rendering original findings, with uncertain groups kept separate. Add synthetic quality-regression cases for severity downgrades and distinct nearby defects.
  - Timing: Fix now, before relying on multi-lane results in an agentic loop. Confirmed with injected synthetic judge responses: bug became nit with `mode=merged` and no bug-policy failure; two distinct nearby findings became one with `mode=merged`. Runtime frequency is unmeasured.

- [x] R2 [P1] Forward the lane tool policy into the reusable judge job
  - Where: `.github/workflows/pr-review.yml:274-297` (judge action inputs), `src/or_pr_review/cli.py:941-953` (stub completeness check).
  - Why it matters: `max_tool_turns` reaches lane jobs but is omitted from the judge job. With two models, tools disabled, and an over-budget diff packed into stubs, lanes see only the stubs. The judge inherits the action's default of 50, so publication skips the explicit tools-disabled partial guard and can publish `clean` plus an authoritative ledger for unseen file contents.
  - Recommendation: Forward `max_tool_turns` to judge and add a reusable-workflow contract test plus a publication regression for stubbed, tool-disabled lanes. Longer term, carry actual review-completeness metadata in artifacts instead of reconstructing it from publisher settings.
  - Timing: Safe to fix now.

- [x] R3 [P1] Apply the lane finding cap by severity and disclose omissions
  - Where: `src/or_pr_review/schema.py:302-315` (`parse_lane_payload`); the normal tool-backed completion path in `src/or_pr_review/harness.py:675-677` does not use response-format schema enforcement.
  - Why it matters: Overflow is sliced to the first 80 raw rows before parsing. A model that follows the per-file sweep order can report nits early and a bug at row 81; the bug is dropped before judging, publication, or ledger accounting. Offline check: 80 nits followed by one bug retained zero bugs and `fail_on=bugs` returned false. Only a log warning records the cap; artifact metadata does not preserve the omitted count. The previous audit fixed cap handling in the judge, not this earlier lane cap.
  - Recommendation: Validate candidates, retain the strongest severities using the shared ranking, and carry the dropped count into the artifact and visible review. Keep incomplete accounting explicit when the discarded tail cannot be validated; add an overflow case with the highest-severity finding last.
  - Timing: Fix now; applies to single-lane and multi-lane review. Frequency on real runs is not measured.

## Medium Cleanup Items

- [ ] R4 [P2] Publish completed matrix lanes from the context they actually reviewed
  - Where: `src/or_pr_review/cli.py:211-219`, `:879-891` (`_finish` re-collects before processing artifacts); `src/or_pr_review/collect.py:405-412` (live-head confirmation); `src/or_pr_review/schema.py:144-195` (artifact context).
  - Why it matters: A new push after full-PR lanes finish causes the judge's fresh collection to reject the old pinned head before it can use the completed findings. The intended stale-review publication path later in `_finish` is never reached. Collection also reloads the loop ledger rather than carrying the exact prior state used by the lanes. In an agentic loop with frequent pushes, this discards usable paid review work and makes publication depend on unrelated fresh reads. Offline control-flow check confirmed that a collection head-change error prevents any merge attempt despite two completed lanes.
  - Recommendation: Persist a bounded review context with the matrix artifacts (reviewed commit, diff/scope/completeness, prior ledger generation/round and context identity). Validate all lanes against it; use the live head only to decide whether publication is stale, and withhold authoritative new state when stale. Avoid recollecting the full review input just to publish it.
  - Timing: Fix after the small workflow correction. This is a new failure mode adjacent to the old report's already-fixed wrong-directory recollection issue; no live race was induced.

- [x] R5 [P2] Retain paid judge usage even when its answer cannot be parsed
  - Where: `src/or_pr_review/judge.py:575-607` (`run_llm_judge`), `src/or_pr_review/cli.py:328-366` (`_resolve_issues` fallback), `src/or_pr_review/publish.py:251-270` (`_cost_note`).
  - Why it matters: Judge usage is extracted only after parsing and merging succeed. A paid response with invalid JSON raises first; the fallback sets cost to null and ran to false. The review can therefore show a complete-looking total that omits the paid judge. Offline check: two $0.10 lanes plus a malformed $0.03 judge response displayed `$0.20`, with no incomplete-cost note.
  - Recommendation: Capture usage immediately after the response arrives and preserve it independently of merge success. Track attempted, completed, and accepted judge work separately; flag totals as incomplete if an attempted call's charge is unknown.
  - Timing: Safe to fix now; important before using posted totals for model/value comparisons.

- [ ] R6 [P2] Give lane benchmark findings explicit matching and duplicate accounting
  - Where: `src/or_pr_review/bench.py:468-508` (`match_finding`, `score_run`), `:178-194` (`RunScore.precision`, `noise`).
  - Why it matters: Every finding can match every same-file keyword label, including a keyword mentioned only to say that behavior is correct. Duplicate findings each count as true positives and add no noise. Offline check: one finding reporting rounding while explicitly saying empty-input handling is correct earned 2/2 recall for separate rounding/empty-input labels; four copies earned 4/4 precision and 0/4 noise. This can reward the same over-consolidation and repetitive output the real loop needs to measure accurately.
  - Recommendation: Bring the lane scorer in line with the judge scorer's one-to-one assignment and separate duplicate rate. Treat ambiguous keyword matches as needing adjudication; pin adversarial scoring examples. Keep unmatched candidates separate rather than automatically calling them false positives. Measure downstream adjudication cost as well as defect recall.
  - Timing: Fix before the next prompt/model adoption decision. This demonstrates scorer behavior, not that historical published results are all wrong; re-score saved runs where relevant.

- [ ] R7 [P2] Bind lane benchmark results to their actual fixture and run configuration
  - Where: `src/or_pr_review/bench.py:527-623` (`_cmd_run` writes only `lane.to_dict()`), `:629-652` (`_cmd_score`); compare the richer judge benchmark record at `:1060-1073`.
  - Why it matters: Lane output omits fixture/prompt hashes, harness revision, effort, tool/deadline settings, and requested provider policy. Scoring accepts any successful `run-*.json` against the supplied fixture. An older output can be scored against regenerated labels/checkout, and experiment directories or handwritten notes become the only source of which configuration produced the result. The judge benchmark already rejects fixture hash mismatches; the primary lane benchmark does not.
  - Recommendation: Add a versioned lane-benchmark envelope or companion manifest containing content hashes and resolved settings; validate it during scoring. Keep production `LaneResult` separate from benchmark metadata. Preserve legacy reads with a visible provenance warning.
  - Timing: Safe to add before more paid comparison runs; do not rerun existing paid experiments just to fill metadata.

## Nice-to-Have Polish

None proposed. Retained compatibility aliases, strict credential-file refusal, the reserved persona input, and previously accepted deferrals remain closed. Module length by itself is not a new finding.

## Validation and prior-report comparison

- Existing tests: 456 passed, 2 skipped (16.42 seconds), using local Python and cache-disabled pytest.
- CI-scoped Ruff lint passes; formatting check reports 50 files already formatted.
- An initial broad lint command also traversed unrelated local `.codex/worktrees` and reported 16 errors there. Those are excluded from this checkout's findings; the actual CI scope is clean.
- Prior report located following user clarification. Earlier root/history search found no report because it is intentionally a scan-local file in another worktree.
- No live provider requests or GitHub writes performed.
- Additional checks ran from stdin with synthetic data, injected model responses, mocked GitHub publication, and no test-file additions. R1/R2/R3/R5/R6 have concrete offline result checks; R4 combines a control-flow check with the live-head guard trace; R7 is a producer/consumer contract inspection. No live workflow run or provider quality comparison was performed.
- `git diff --check` passes. Git status preserves the existing untracked `.codex/` and adds only this report; no source edits, commits, or pushes.
- Two tests were skipped in this Windows run; Linux CI was not rerun by this audit.
- Scope inspected: action/reusable workflow wiring, collection and triage, model/tool loop, workspace/tool safety, schema and merge/judge boundaries, ledger/publication, benchmark runners/scorers, tests and project documentation. This is a cleanup/correctness audit, not a claim of exhaustive security validation.

## High-level assessment and hobby-sized next steps

Building this was a reasonable choice. The useful custom work is the review contract: bounded read-only investigation, preservation of minority findings, commit-aware follow-up, agent disputes, and measurable model/provider choices. The source has no runtime package dependencies, and the existing tests and recorded negative experiments show deliberate engineering. The maintenance burden is concentrated in provider compatibility and the cross-job state contract. A framework replacement should earn its keep by eliminating that burden without weakening those behaviors.

The objective should be high defect recall and correct convergence at a tolerable total cost. A cheap candidate that the fixing agent can quickly disprove can be worthwhile; a plausible false report that causes a bad edit is expensive. Count reviewer calls, fixing-agent work, verification, retries, abandoned runs, and introduced regressions. Model token prices and successful-run recall alone do not determine best value.

Recommended sequence:

1. Fix the small correctness holes first: R2, R3, and R1. Keep canonical findings as data and make preservation properties executable. A merge model should organize findings without being able to silently downgrade or rewrite them away.
2. Strengthen existing evaluation infrastructure, coordinating with the separate benchmark projects instead of creating another competing benchmark system here. Fix R6/R7, then use a small frozen set of representative PRs to evaluate initial review -> proposed fix -> verification. Include correct fixes, incomplete fixes, valid rebuttals, and a new push arriving mid-review. Preserve a held-out evaluation split outside this public action repository. Track distinct bugs fixed, missed bugs, unnecessary edits, new regressions, rounds to convergence, total spend, and latency. This follows the distinction between an agent's output and the actual environment outcome discussed in [Anthropic's agent evaluation guidance](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).
3. Establish one model as the default measured baseline; compare one stronger reviewer with one cheaper reviewer and a two-reviewer configuration under comparable total budgets. Promote a second lane for the additional valid defects it finds that the first misses. Require evidence that it improves the full loop rather than assuming vendor diversity means independent errors. Do not remove uncertain minority findings merely to reduce comment volume.
4. Improve evidence access before adding personas or orchestration. A bounded on-demand diff/base-version reader for stubbed files would preserve removed behavior currently reduced to counts; targeted caller/test context can save repeated broad searches. Measure cache behavior as part of each configuration: stable prefixes and explicit caching where required can materially affect costs. [OpenRouter documents provider-dependent caching and sticky routing](https://openrouter.ai/docs/guides/best-practices/prompt-caching); it already provides automatic stickiness in some circumstances, so missing a custom routing layer is not itself a defect.
5. Make ordinary use boring: one default lane, a few benchmarked overrides, bounded budgets, durable review context, and deterministic publication. Keep experimental lane/persona choices in benchmark configuration until they show repeatable benefit. Avoid turning this hobby into a general-purpose agent platform. Broader frameworks or native review agents remain useful comparison baselines, not automatic replacements.

A practical first milestone is a saved, reproducible experiment showing that a particular configuration fixes at least as many important defects with lower total loop cost and no increase in bad edits. The current scan does not establish which model achieves that; no new paid benchmark was run.
