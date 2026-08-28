# Recall bench

Offline, repeatable measurement of lane recall/precision — no live PR, no
GitHub posting. Replays a frozen review fixture through `run_lane` and scores
the findings against curated golden labels.

Three kinds of fixture:

- **Planted** (`bench/fixtures/planted-mini/`, committed): a small synthetic
  project whose diff contains known defects at every severity. Objective
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
    --model x-ai/grok-4.6 --out /tmp/bench-out
python -m or_pr_review.bench score bench/fixtures/planted-mini /tmp/bench-out
```

Lanes are nondeterministic — `run` defaults to 3 runs, and `score` prints a
mean row across the successful runs. `run` clears stale `run-*.json` files
from `--out` first (leftovers would silently mix experiments), exits
non-zero when any lane fails, and defaults `--effort` to empty to match the
shipped action (pass `--effort high` explicitly to mirror a caller that sets
it). `score` reports per-severity and per-context recall (detection,
regardless of the severity the finding reported), a `sev-agree` column
(matched labels hit at the label's own severity), adjudicated precision, a
`noise` column (adjudicated-false plus unadjudicated findings — the
oversensitivity headline), per-label detection frequency across runs, the
labels every run missed, and each run's non-label findings tagged with the
adjudication verdict that fired or `UNADJUDICATED` for triage.

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
PR advances mid-capture — just re-run), records `max_diff_kb` (default 600,
the org value) so the replay embeds the same truncated diff production
would, and refuses `--out` under committed `bench/fixtures/` unless
`--allow-committed` is passed.

Then curate `labels.json` from the PR's adjudicated findings (every
reviewer's validated true positives — including findings the lane under test
originally missed).
