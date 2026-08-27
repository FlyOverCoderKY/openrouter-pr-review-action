# Recall bench

Offline, repeatable measurement of lane recall/precision — no live PR, no
GitHub posting. Replays a frozen review fixture through `run_lane` and scores
the findings against curated golden labels.

Two tiers of fixture:

- **Planted** (`bench/fixtures/planted-mini/`, committed): a small synthetic
  project whose diff contains known defects at every severity. Objective
  ground truth, pennies per run, minutes to iterate. The unit test of
  promptcraft.
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
it). `score` reports per-severity recall (detection, regardless of the
severity the finding reported), a `sev-agree` column (matched labels hit at
the label's own severity), precision, the labels every run missed, and each
run's unmatched findings (triage those: a consistent unmatched finding may
be a new true positive worth adding as a label, or noise worth a prompt
tweak).

## Label format

A label matches a finding when the finding cites the label's `file` (or the
label has no file) and any `keywords` regex matches the finding's title+body,
case-insensitively. Pick keyword fragments a correct finding could not avoid
mentioning; give several alternates per label.

```json
{"id": "B1", "severity": "bug", "file": "calc.py",
 "title": "Stale 2027 cap", "keywords": ["same value as (the )?2026", "8[_,]?300.{0,40}2027"]}
```

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
