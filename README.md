# openrouter-pr-review-action

Public MIT GitHub Action: [OpenRouter](https://openrouter.ai) pull-request reviews with a **custom thin harness** and an optional **multi-model bake-off**.

Author: **Nathan (FlyOverCoderKY) / RetireGolden, LLC**.

Sibling to [`grok-pr-review-action`](https://github.com/FlyOverCoderKY/grok-pr-review-action) (Grok CLI + `XAI_API_KEY` against `api.x.ai`). This action is an independent stack: it talks **only** to OpenRouter with `OPENROUTER_API_KEY`. It never reads `XAI_API_KEY`, never calls `api.x.ai`, and never downloads or invokes the Grok CLI.

First production use is a **single Grok 4.6 lane** beside the existing Grok action so you can compare the whole stacks (cost, speed, findings). If this one hits parity, turn the old action off and add more OpenRouter models via `models`.

## What it does

- One invocation = one review. Callers own concurrency and merge gating.
- `models` is a comma-separated list of OpenRouter slugs. **List length is the lane count** (hard-capped at **4**; the action fails clearly if you ask for more).
- **One lane:** the judge is skipped. That lane’s structured findings are posted directly. No extra OpenRouter call, no merge/de-dupe, no invented cross-model attribution. The finding can still name the model that produced it.
- **Two or more lanes:** parallel review lanes (same prompt on every lane in v1), then an OpenRouter **judge** merges and de-dupes and posts **one** GitHub review. Attribution looks like:

  ```text
  Issue 1 - Missing auth check (identified by x-ai/grok-4.6 and anthropic/claude-sonnet-4.6)
  ```

  then the issue body.

- Prefer a **setup → matrix lanes → judge** workflow so wall clock is roughly the slowest lane + judge, not the sum. GitHub **bills the minutes in parallel**. A single-job `role: all` run can still fan lanes in-process; that job is billed as one runner.
- If a **lane** fails, that lane fail-opens and the judge (or the single-lane poster) continues on whatever structured results arrived. Operational or **schema** errors of the action itself fail the job.

Personas / specialized prompts are **out of v1**. The `persona` input is a reserved unused hook so a later persona feature does not require a rewrite. A future **single-persona** run should skip the judge the same way (one reviewer = no judge).

## Auth

Real auth is **`OPENROUTER_API_KEY` against `https://openrouter.ai/api/v1/chat/completions` only**.

1. Create an API key at [openrouter.ai](https://openrouter.ai).
2. Store it as the repository secret `OPENROUTER_API_KEY`.
3. Pass it as a **job environment variable**, not as an action input (so it stays out of input logs):

```yaml
env:
  OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

The action fails closed if the key is empty (when a lane or the multi-lane judge needs it), never prints the key, and never documents or accepts `XAI_API_KEY`.

## Privacy

This action uses OpenRouter as an external processor. The prompt sends PR metadata, the selected diff, and `custom_instructions` to OpenRouter, which routes them to the **upstream model provider** for each slug. When a lane uses `read_file`, `grep`, or `list_dir`, those paths and file contents can also be sent as model context.

Do not put secrets, credentials, regulated personal data, or unrelated confidential material in PR text, diffs, repository files, or `custom_instructions`. Obtain organizational approval before enabling the action on private or regulated repositories.

## Untrusted pull requests

PR title, body, diffs, and repository files are treated as **untrusted data**. The action does **not** execute PR code. Before the model runs, tracked files from the reviewed commit are materialized into an inert, size-bounded checkout. Tools are read-only (`read_file`, `grep`, `list_dir`). There is no shell, no writes, no network except OpenRouter, no subagents, and no memory. Secret-like paths (`.env`, keys, credential files) are refused.

The checkout must contain the reviewed commit object (`fetch-depth: 0` is the safest setup). GitHub does not give repository secrets or a write token to ordinary `pull_request` workflows from public forks. Do not work around that by blindly checking out and executing fork code under `pull_request_target`.

The default first-pass tool budget is **50** read-only rounds (`max_tool_turns`), matching the sibling Grok action's default `max_turns`. The review prompt tells the model to use those tools for **blast radius** — filename-inventory tests, README / code-map docs, and sibling CI files — not just the embedded diff. A workflow-only PR can still break a test that requires every `.github/workflows/*.yml` to be listed in docs. Follow-up jobs may pass a lower budget (sibling callers often use `30`).

## Copy-paste: one-lane Grok via OpenRouter

First-pass + latest-commit follow-up. The default `models` value is one slug (`x-ai/grok-4.6`), so the **judge does not run**.

```yaml
name: OpenRouter PR review

on:
  pull_request:
    types: [opened, reopened, ready_for_review, synchronize]

jobs:
  first-pass:
    if: github.event.action != 'synchronize'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
    concurrency:
      group: or-review-first-pass-${{ github.repository }}-${{ github.event.pull_request.number }}
      cancel-in-progress: true
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          persist-credentials: false
          fetch-depth: 0
          ref: ${{ github.event.pull_request.head.sha }}

      - uses: FlyOverCoderKY/openrouter-pr-review-action@main
        with:
          github_token: ${{ github.token }}
          pr_number: ${{ github.event.pull_request.number }}
          models: x-ai/grok-4.6
          review_scope: full-pr
          review_mode: initial
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}

  follow-up:
    if: github.event.action == 'synchronize'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
    concurrency:
      group: or-review-follow-up-${{ github.repository }}-${{ github.event.pull_request.number }}
      cancel-in-progress: true
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          persist-credentials: false
          fetch-depth: 0
          ref: ${{ github.event.pull_request.head.sha }}

      - uses: FlyOverCoderKY/openrouter-pr-review-action@main
        with:
          github_token: ${{ github.token }}
          pr_number: ${{ github.event.pull_request.number }}
          models: x-ai/grok-4.6
          review_scope: latest-commit
          review_mode: verify
          max_tool_turns: "30"
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

Pin `@main` only until you choose a commit SHA. Do not put first-pass and follow-up in one concurrency group that `synchronize` can cancel.

`latest-commit` never silently falls back to the full PR diff. If `before...after` is missing or the compare fails, the action embeds the **single latest head commit** and says so. A truncated diff posts a visible **partial** verdict and is never treated as clean.

The initial round is the exhaustive pass: the prompt requires a per-file, all-severity sweep (bug, risk, **and** nit), and each coverage entry claims a completed sweep of that file. From verify round 2 onward a **severity floor** applies: carried `nit` findings are retired — stated visibly in the round's resolution section, never silently — so follow-up rounds track the bug/risk backlog to convergence instead of re-adjudicating nits forever.

## Copy-paste: multi-model bake-off

Use the reusable workflow so lanes are **separate GitHub jobs**. Wall clock ≈ slowest lane + judge. GitHub bills those minutes **in parallel**.

```yaml
name: OpenRouter bake-off

on:
  pull_request:
    types: [opened, reopened, ready_for_review]

jobs:
  review:
    uses: FlyOverCoderKY/openrouter-pr-review-action/.github/workflows/pr-review.yml@main
    permissions:
      contents: read
      pull-requests: write
    with:
      models: x-ai/grok-4.6,anthropic/claude-sonnet-4.6
      review_scope: full-pr
      review_mode: initial
      # Optional. Default is google/gemini-3.1-flash-lite (merge only).
      # judge_model: openai/gpt-4.1-nano
    secrets:
      OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

Two or more slugs **require** a judge. The default judge is `google/gemini-3.1-flash-lite` (verified live on OpenRouter). It is merge/de-dupe of already-structured findings plus JSON schema — not a second reviewer. Thinking/reasoning is pinned to `minimal`.

Alternatives (do not change the default unless you mean to):

| Slug | When to use |
| --- | --- |
| `openai/gpt-4.1-nano` | Cheaper/faster if you want to trade merge quality |
| `anthropic/claude-haiku-4.5` | Upgrade if the Flash Lite judge is dropping duplicates |

A judge schema mismatch fails the job (fail-closed).

## Recommended caller concurrency

This action runs **one review per invocation**. It does not implement workflow concurrency or org-specific merge gates (those belong in your reusable caller).

Use **separate first-pass and follow-up jobs**, or distinct concurrency groups, so a `synchronize` run cannot cancel an in-progress `full-pr` review.

## Inputs

| Input | Default | Notes |
| --- | --- | --- |
| `role` | `all` | `all` (collect + lanes + optional judge + post) \| `setup` (parse matrix) \| `lane` \| `judge` |
| `models` | `x-ai/grok-4.6` | Comma-separated OpenRouter slugs. Length = lane count. Cap 4. |
| `judge_model` | `google/gemini-3.1-flash-lite` | Independent of `models`. Used only when two or more lanes are configured. |
| `judge_needed` | _empty_ | `true` / `false` override. Empty infers from `models` length. |
| `github_token` | `${{ github.token }}` | Needs `pull-requests: write` to post the review. |
| `github_timeout_seconds` | `120` | Per-operation `gh` timeout; 1–600. |
| `pr_number` | PR that triggered the workflow | Required for `workflow_dispatch`. |
| `fail_on` | `never` | Finding policy: `never` \| `bugs` \| `any`. Operational/schema errors always fail. |
| `roast_level` | `professional` | `professional` \| `playful`. |
| `custom_instructions` | _empty_ | Extra prompt text, max 16,000 UTF-8 bytes. Never put secrets here. |
| `status_comments` | `true` | Live status comment on the PR. |
| `max_diff_kb` | `300` | Embedded diff cap. Truncation ⇒ `partial`, never clean. |
| `review_scope` | `full-pr` | `full-pr` \| `latest-commit`. Initial rounds require `full-pr`. |
| `review_mode` | `auto` | `auto` (opened = initial, synchronize = verify) \| `initial` \| `verify`. |
| `effort` | _empty_ | Optional OpenRouter reasoning effort for **review lanes**. |
| `max_tool_turns` | `50` | Read-only tool rounds against the inert checkout. `0` disables tools. First-pass default matches the sibling Grok `max_turns`. Follow-up jobs may pass `30`. |
| `bot_login` | `github-actions[bot]` | Identity the action posts reviews as. Review-loop ledger state is only trusted from this login. Change it when `github_token` is a PAT or App token. |
| `persona` | _empty_ | **Reserved, unused in v1.** Future single-persona runs should skip the judge. |

## Outputs

| Output | Meaning |
| --- | --- |
| `verdict` | `clean` \| `issues` \| `partial` \| `error` |
| `issue_count` | Open findings after this round (carried-over plus new; nits retired by the severity floor excluded) |
| `bug_count` | Open bug-severity findings after this round |
| `round` | Review-loop round number this run performed |
| `review_url` | Posted GitHub review |
| `models_json` | Parsed slug array (`setup` / `all`) |
| `matrix` | `[{index, model}, …]` for a GitHub Actions matrix |
| `lane_count` | Number of lanes |
| `judge_needed` | `true` when two or more lanes require the judge |
| `judge_model` | Judge slug (ignored on a single lane) |
| `lane_file` | Written lane JSON (`role=lane`) |

## Local tests

CI on this repo does not need a live `OPENROUTER_API_KEY`.

For measuring review recall against fixtures with known defects (which DOES
spend tokens and needs `OPENROUTER_API_KEY`), see the offline bench in
[bench/README.md](bench/README.md).

```bash
python3 -m pip install -e '.[dev]'
pytest
```

## License

[MIT](LICENSE). Copyright (c) 2026 Fly Over Coder.
