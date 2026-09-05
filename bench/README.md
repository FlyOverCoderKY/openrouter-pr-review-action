# Recall bench

Offline, repeatable measurement of lane recall/precision — no live PR, no
GitHub posting. Replays a frozen review fixture through `run_lane` and scores
the findings against curated golden labels.

Three kinds of fixture:

- **Planted** (`bench/fixtures/planted-mini/`, committed): a small synthetic
  project seeded with known defects at every severity and context stratum —
  most visible in the diff, but some findable only by reading a changed file
  in full (`file`) or following callers outside the diff (`repo`). Objective
  ground truth, pennies per run, minutes to iterate. The unit test of
  promptcraft.
- **Clean twin** (`bench/fixtures/planted-mini-clean/`, committed): the same
  pull request done correctly, zero labels — measures oversensitivity. Run it
  alongside the planted fixture whenever a prompt or validator changes.
- **Real-PR replays** (`bench/fixtures-local/`, gitignored): captured from
  live PRs with `bench/capture.py`, labeled from the adjudicated union of all
  reviewers' validated findings. The integration test. These contain private
  repository code and must never be committed here. A replay runs cache-cold:
  a dense-PR fixture costs its full prompt-token weight per run.

## Run and score

```bash
export OPENROUTER_API_KEY=...   # `run` spends real tokens; `score` is offline
python -m or_pr_review.bench run bench/fixtures/planted-mini \
    --model x-ai/grok-4.6 --out /tmp/bench-out --allow-spend
python -m or_pr_review.bench score bench/fixtures/planted-mini /tmp/bench-out
```

To prepare an Astra standard-versus-Flex comparison, add
`--service-tier default` or `--service-tier flex` to otherwise identical
**opt-in live** runs. This flag is intentionally benchmark-only and does not
change the action's defaults. It requests OpenRouter's top-level
`service_tier`; it does not prove that tier was served. The run artifact and
the aggregate progress checkpoint record the requested tier separately from
the distinct response-reported served tiers, the number of observed tier
responses, completeness, and confirmation (only every served value matching
the request confirms it). Missing, `null`, mixed, or interrupted response
telemetry must not be treated as confirmation of Flex.

`run` makes no request unless both `--allow-spend` is present and
`OPENROUTER_API_KEY` is set. Lanes are nondeterministic — `run` defaults to 3
runs, and `score` prints a mean row across the successful runs. During each
paid lane, `run` atomically
writes an aggregate-only `progress-N.json` checkpoint containing elapsed time,
usage, request/tool counts, provider, and provider-reported cost; it contains no
prompt, code, findings, paths, or credentials. A completed `run-N.json` replaces
that checkpoint. `run` clears stale `run-*.json`, `progress-*.json`, and
`progress-*.tmp` files from `--out` first (leftovers would mix experiments), exits
non-zero when any lane fails, and defaults `--effort` to empty to match the
shipped action (pass `--effort high` explicitly to mirror a caller that sets
it). `score` reports per-severity and per-context recall (detection,
regardless of the severity the finding reported), a `sev-agree` column
(matched labels hit at the label's own severity), adjudicated precision, a
`noise` column (adjudicated-false plus unadjudicated findings — the
oversensitivity headline), per-label detection frequency across runs, the
labels every run missed, and each run's non-label findings tagged with the
adjudication verdict that fired or `UNADJUDICATED` for triage.

Completed `run-N.json` files contain finding bodies and file paths. For real-PR
fixtures, keep `--out` outside this repository as well as the fixture itself.

Cost telemetry is conservative: `known_cost_usd` retains valid per-response
spend for budgeting, while `cost_usd` is present only when every attempted
request reported a valid cost. A timeout, HTTP error, or response without a
valid cost leaves the total incomplete; it is never silently counted as $0.
The focused service-tier and accounting tests use injected synthetic responses
only. They make no provider call and require no OpenRouter credentials.

## Label format

A label matches a finding when the finding cites the label's `file` (or the
label has no file) and any `keywords` regex matches the finding's title+body,
case-insensitively. Pick keyword fragments a correct finding could not avoid
mentioning; give several alternates per label.

```json
{"id": "B1", "severity": "bug", "file": "calc.py", "context": "diff",
 "title": "Stale 2027 cap", "keywords": ["same value as (the )?2026", "8[_,]?300.{0,40}2027"]}
```

`context` records the minimum context needed to find the plant — `diff`
(visible in the embedded diff), `file` (requires reading the changed file
beyond its hunks), or `repo` (requires tool use outside the diff). `score`
reports recall per stratum; a prompt or validation change must not regress
any stratum, because aggregate recall can hide local-context loss.

## Adjudications and the clean twin

Unmatched findings are three-way classified, never auto-counted as false:
a fixture may carry an `adjudications.json` of curated verdicts
(`true_positive_unlabeled` counts toward precision, `false_positive`
against it; same file+keywords matching as labels), and anything else is
reported as `UNADJUDICATED` for triage — promote it to a label, adjudicate
it, or treat it as a prompt problem.

`planted-mini-clean` is the unmutated twin: the same pull request done
correctly, zero labels. Every finding a lane reports against it is a noise
candidate, which makes oversensitivity (e.g. a prompt change that invites
padding) measurable instead of invisible. Run it alongside planted-mini
whenever a prompt or validator changes.

Fixture checkouts and diffs are regenerated only through
`bench/fixtures/generate_planted.py`; a unit test pins the committed trees
to that script, so edit the script, not the checkout.

## Run-count protocol

At least 3 runs for quick screening; at least 5 per configuration for a
decision that changes production behavior. Compare means and per-label
detection frequency, not single runs.

## Capturing a real PR

```bash
python bench/capture.py --repo RetireGolden/RetireGolden --pr 331 \
    --clone ~/src/RetireGolden --out bench/fixtures-local/rg-331
```

Capture pins metadata, diff, and checkout to one PR head (it aborts if the
PR advances mid-capture — just re-run), records `max_diff_kb` (default 300,
matching the action; pass the deployed repository override when applicable)
so the replay embeds the same truncated diff production would, and refuses
`--out` under committed `bench/fixtures/` unless
`--allow-committed` is passed.

Then curate `labels.json` from the PR's adjudicated findings (every
reviewer's validated true positives — including findings the lane under test
originally missed).

## Merge-judge A/B and A/B/C benchmark

`bench/judge-fixtures/` is a separate, fully synthetic test suite for the
multi-lane merge judge. It does not contain repository code or held-out task
data. The committed sets exercise:

- duplicate reports plus complementary valid findings;
- distinct, superficially conflicting defects at the same location;
- unsupported review-tool/checkout failures that must remain diagnostics;
- a recall-critical finding reported only by the minority C lane; and
- a clean verdict when every lane item is an environmental diagnostic.

Scoring is deterministic and offline. A result JSON needs only an `issues`
array; live-run metadata is optional:

```bash
python -m or_pr_review.bench judge-score \
  bench/judge-fixtures/abc-conflicts-minority.json saved-result.json
```

The score reports issue-level precision and recall using one-to-one matching,
duplicate rate, minority/critical recall, and verdict correctness. One broad
merged row cannot earn recall for several expected issues. Results produced
from a different fixture revision are rejected by SHA-256 instead of being
silently mixed.

### Opt-in live model comparison

`run` and `judge-run` are the paid benchmark commands; `judge-run` is the only
paid judge-benchmark command. Neither makes a request unless **both**
`--allow-spend` is present and `OPENROUTER_API_KEY` is set.
Use a new output directory per experiment; existing result filenames are
never overwritten. The complete model/run output matrix is collision-checked
before the first request, and `--runs` has a hard safety cap of 20.

```bash
export OPENROUTER_API_KEY=...
python -m or_pr_review.bench judge-run \
  bench/judge-fixtures/abc-conflicts-minority.json \
  --models google/gemini-3.1-flash-lite,openai/gpt-5.6-luna \
  --runs 5 --out /tmp/judge-abc-20260829 --allow-spend
python -m or_pr_review.bench judge-score \
  bench/judge-fixtures/abc-conflicts-minority.json \
  /tmp/judge-abc-20260829
```

Run every candidate against all three fixtures, at least 3 runs for a quick
screen and 5 before changing production. The saved record includes model,
fixture and prompt hashes, harness version, UTC start, latency, judge repair
mode, OpenRouter response/provider identifiers, token counts, and reported
cost; it never stores the API key or raw response. Compare final metrics
and repair mode: two models can both finish at 100% after recall-safe repair,
while the one that reaches `merged` without splits/restores is the better
clerical judge.

The production default is `openai/gpt-5.6-luna`, adopted from the 2026-08-29
five-run-per-fixture decision screen recorded in [RESULTS.md](RESULTS.md).
The adoption screen ran exactly these models:

- `google/gemini-3.1-flash-lite` — the former default and benchmark baseline;
- `openai/gpt-5.6-luna` — the adopted default.

Potential follow-up candidates that have **not** been benchmarked are
`openai/gpt-5.4-mini` (same-vendor control) and
`anthropic/claude-haiku-4.5` (cross-vendor control). Do not treat either as a
production-quality alternative until it completes the protocol above.

OpenRouter's catalogue changes, so re-check availability and structured-output
support before spending on a later experiment. The synthetic acceptance bar
is intentionally strict: 100% recall (including critical recall), 100%
precision, 0% duplicates, and 100% verdict accuracy. Cost and latency break
ties; they do not excuse a recall miss.
