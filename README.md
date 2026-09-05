# openrouter-pr-review-action

An MIT-licensed GitHub Action that reviews pull requests using one or more models through [OpenRouter](https://openrouter.ai). It reads the diff and repository context, posts structured findings, and tracks unresolved findings across follow-up reviews. It works with human authors and coding agents; it does not make fixes or merge PRs.

Author: **Nathan (FlyOverCoderKY) / RetireGolden, LLC**.

**Start here:** [Setup](#setup) · [Choose models](#choosing-models) · [Copy-paste workflow](#copy-paste-one-lane-grok-via-openrouter) · [Agentic loops and merge gates](#agentic-loops-and-merge-gates) · [Troubleshooting](#troubleshooting) · [Inputs](#inputs)

## Setup

1. Enable GitHub Actions in the consuming repository and allow this action under any organization action restrictions. The examples use GitHub-hosted `ubuntu-latest`. A custom runner needs Bash, Git, GitHub CLI (`gh`), Python 3.11+, and access to GitHub and OpenRouter; no application dependency installation is needed.
2. Create an OpenRouter key with credit available and save it under **Settings → Secrets and variables → Actions → New repository secret**, named `OPENROUTER_API_KEY`. An organization secret also works if this repository has access. Keep the value out of chat and committed files.
3. Save the [one-lane workflow below](#copy-paste-one-lane-grok-via-openrouter) as `.github/workflows/openrouter-review.yml`. It includes initial and follow-up reviews, checkout, permissions, and separate concurrency groups. The examples pin this action to a full commit SHA; update that pin deliberately when adopting fixes.
4. Choose a model using the [benchmark guidance](#choosing-models). Start with one lane to keep setup and cost simple. API usage is paid through your OpenRouter account; GitHub runner usage is separate. Tool/time budgets are limits on work, not a dollar spending cap.
5. Open a small, non-draft PR from a branch in the same repository. Check that the **first-pass** job posts a review with the expected model and head SHA. Wait for it to finish, then push a change and verify that **follow-up** posts the next round. This smoke test makes paid model calls.

The examples intentionally skip drafts, fork PRs, and Dependabot PRs. They are a setup for trusted same-repository contributors. Fork review requires a separately designed trusted workflow; changing the event to `pull_request_target` alone is not a safe installation fix. See [Untrusted pull requests](#untrusted-pull-requests).

For a PR already open when you install the workflow, trigger an initial event first (for example, mark a draft ready for review, or reopen an eligible PR). A push alone starts a verification round and cannot bootstrap a missing initial review. If the initial run is partial or fails, resolve its cause and obtain a complete initial review before relying on follow-ups.

### Ask your coding agent to install it

Copy this prompt into your coding agent:

```text
Set up FlyOverCoderKY/openrouter-pr-review-action in this repository. Read its
README and action.yml at the exact revision you will use, plus this repository's
agent instructions and existing workflows. Pin the action to a full commit SHA.
Use the README's same-repository, non-draft initial + follow-up workflow, keeping
the two concurrency groups separate. Use ubuntu-latest, a 25-minute job timeout,
full-depth checkout at the PR head, persist-credentials: false, contents: read,
and pull-requests: write. Do not install or execute this repository's code in
the review job. Start with one model; consult the linked benchmark and explain
the choice and its limitations. Keep fail_on: never for the initial rollout.
Use the OPENROUTER_API_KEY Actions secret, never a literal key. Tell me if the
secret or repository settings need my attention; do not request the key in chat.
Preserve existing CI and merge rules, check whether an existing review bot would
duplicate this one, and explain any GITHUB_TOKEN-trigger limitation for my agents.
Validate the workflow without calling a model, then describe how to smoke-test
both review rounds and what paid API usage that test will trigger. Do not enable
automatic merging or treat job success as proof of a clean review.
```

## Choosing models

Use the [Review Benchmark leaderboard](https://bench.flyovercoder.com/leaderboard) for model recommendations and measured quality, completion, latency, and cost. Read the [methodology](https://bench.flyovercoder.com/methodology) and each result's status and configuration alongside its rank. The board is a small, harness-specific comparison; AI-adjudicated results are provisional, and a model's position does not certify its performance on every repository.

- **One reviewer:** set `models` to one OpenRouter slug. The default is `x-ai/grok-4.6`; it is a configuration default, not a claim that it is always the best value.
- **Multiple reviewers:** supply up to four comma-separated slugs. Each lane reviews the change, adding API cost. A strong individual ranking does not establish that two models catch different defects; assess the extra useful findings before keeping a second lane.
- **Judge:** `judge_model` combines reviewer findings. Choose it separately from the reviewers; its merge benchmark answers a different question from the review leaderboard. The default and rationale are [below](#copy-paste-multi-model-bake-off).

Translate a board recommendation into the exact OpenRouter model slug in `models`, and check its availability in your account. Keep the chosen slug and action SHA explicit in your workflow so a later comparison has a known starting point. The README links to the evolving results instead of maintaining a second ranking table.

This project is independent of its sibling [`grok-pr-review-action`](https://github.com/FlyOverCoderKY/grok-pr-review-action). You do not need that action, the Grok CLI, or an `XAI_API_KEY` to use this one.

## What it does

- One invocation = one review. Callers own concurrency and merge gating.
- `models` is a comma-separated list of OpenRouter slugs. **List length is the lane count** (hard-capped at **4**; the action fails clearly if you ask for more).
- **One lane:** by default the judge is skipped and that lane’s structured findings are posted directly. Set `judge_needed: true` to force the configured judge for a single lane. The finding can still name the model that produced it.
- **Two or more lanes:** parallel review lanes (same prompt on every lane in v1), then an OpenRouter **judge** union-merges them into **one** GitHub review when the shared job budget leaves a safe judge window. Every input finding is identity-tracked, the judge must account for every id, only duplicates whose source location, title and evidence agree may merge, and any unaccounted or over-broadly merged finding is deterministically restored verbatim before the shared 80-finding publishing cap is applied. A judge transport/schema failure or an exhausted judge window posts the validated deterministic union and labels that degradation in the review instead of discarding completed lanes. If a repaired or fallback union exceeds the cap, its visible mode reports how many lower-severity findings were omitted. Attribution looks like:

  ```text
  #### 🔴 Issue 1 — Missing auth check

  `src/api.py:42` · `bug` · identified by x-ai/grok-4.6 and anthropic/claude-sonnet-4.6
  ```

  then the issue body (severity emoji: 🔴 bug / 🟠 risk / 🔵 nit).

- Prefer a **setup → matrix lanes → judge** workflow so wall clock is roughly the slowest lane + judge, not the sum. GitHub **bills the minutes in parallel**. A single-job `role: all` run can still fan lanes in-process; that job is billed as one runner.
- If a **lane** fails, that lane fail-opens and the judge (or the single-lane poster) continues on whatever structured results arrived. A lane that reports only temporary checkout/tool-access failures produces a visible **partial** review, never an unmarked clean pass. Action-wide collection, input, artifact, and posting errors still fail the job; judge transport/schema errors preserve validated lane findings via a visibly labeled deterministic union, subject to the documented publishing cap.

Persona lanes are **out of v1** (path-scoped guidance is available via `path_profiles`). The `persona` input is a reserved unused hook so a later persona feature does not require a rewrite. A future **single-persona** run should skip the judge the same way (one reviewer = no judge).

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

This action uses OpenRouter as an external processor. The prompt sends PR metadata, the selected diff, `custom_instructions`, and any matching `path_profiles` instructions to OpenRouter, which routes them to the **upstream model provider** for each slug. When a lane uses `read_file`, `grep`, or `list_dir`, those paths and file contents can also be sent as model context.

Do not put secrets, credentials, regulated personal data, or unrelated confidential material in PR text, diffs, repository files, `custom_instructions`, or `path_profiles`. Obtain organizational approval before enabling the action on private or regulated repositories.

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
    if: >-
      github.event.action != 'synchronize' &&
      github.event.pull_request.draft == false &&
      github.event.pull_request.head.repo.full_name == github.repository &&
      github.event.pull_request.user.login != 'dependabot[bot]' &&
      github.actor != 'dependabot[bot]'
    runs-on: ubuntu-latest
    timeout-minutes: 25
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

      - uses: FlyOverCoderKY/openrouter-pr-review-action@d01d16c4581e2de9110192382637c277587ad5a2
        with:
          github_token: ${{ github.token }}
          pr_number: ${{ github.event.pull_request.number }}
          models: x-ai/grok-4.6
          review_scope: full-pr
          review_mode: initial
          fail_on: never
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}

  follow-up:
    if: >-
      github.event.action == 'synchronize' &&
      github.event.pull_request.draft == false &&
      github.event.pull_request.head.repo.full_name == github.repository &&
      github.event.pull_request.user.login != 'dependabot[bot]' &&
      github.actor != 'dependabot[bot]'
    runs-on: ubuntu-latest
    timeout-minutes: 25
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

      - uses: FlyOverCoderKY/openrouter-pr-review-action@d01d16c4581e2de9110192382637c277587ad5a2
        with:
          github_token: ${{ github.token }}
          pr_number: ${{ github.event.pull_request.number }}
          models: x-ai/grok-4.6
          review_scope: latest-commit
          review_mode: verify
          fail_on: never
          max_tool_turns: "30"
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

The examples pin the review-integrity implementation at `d01d16c4581e2de9110192382637c277587ad5a2`; a branch push does not update existing consumers' pins. Do not put first-pass and follow-up in one concurrency group that `synchronize` can cancel.

`latest-commit` never silently falls back to the full PR diff. If `before...after` is missing or the compare fails, the action embeds the **single latest head commit** and says so. For `full-pr`, if GitHub rejects `gh pr diff` because the patch exceeds its line limit, the action computes the complete `git diff base...head` from the full-depth workflow checkout before applying the normal diff-budget triage. A truncated diff posts a visible **partial** verdict and is never treated as clean.

The initial round is the exhaustive pass: the prompt requires a per-file, all-severity sweep (bug, risk, **and** nit), and each coverage entry claims a completed sweep of that file. From verify round 2 onward a **severity floor** applies: carried `nit` findings are retired — stated visibly in the round's resolution section, never silently — so follow-up rounds track the bug/risk backlog to convergence instead of re-adjudicating nits forever.

## Copy-paste: multi-model bake-off

Use the reusable workflow so lanes are **separate GitHub jobs**. Wall clock ≈ slowest lane + judge. GitHub bills those minutes **in parallel**. This example runs an initial review only. For a complete loop, the one-job example above also accepts multiple slugs; alternatively add a separate reusable-workflow follow-up job with `synchronize`, `review_mode: verify`, `review_scope: latest-commit`, and its own concurrency group.

```yaml
name: OpenRouter bake-off

on:
  pull_request:
    types: [opened, reopened, ready_for_review]

jobs:
  review:
    if: >-
      github.event.pull_request.draft == false &&
      github.event.pull_request.head.repo.full_name == github.repository &&
      github.event.pull_request.user.login != 'dependabot[bot]' &&
      github.actor != 'dependabot[bot]'
    concurrency:
      group: or-review-first-pass-${{ github.repository }}-${{ github.event.pull_request.number }}
      cancel-in-progress: true
    uses: FlyOverCoderKY/openrouter-pr-review-action/.github/workflows/pr-review.yml@d01d16c4581e2de9110192382637c277587ad5a2
    permissions:
      contents: read
      pull-requests: write
    with:
      models: x-ai/grok-4.6,anthropic/claude-sonnet-4.6
      review_scope: full-pr
      review_mode: initial
      # Optional. Default is openai/gpt-5.6-luna (merge only).
      # judge_model: google/gemini-3.1-flash-lite  # measured lower-latency override
    secrets:
      OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

Two or more slugs schedule a judge. When using the direct action, `judge_needed: true` can also force it for one configured lane; the reusable workflow does not expose that override. The default judge is `openai/gpt-5.6-luna`, selected by the repeatable synthetic judge benchmark after it retained 50/50 expected findings with 100% precision, zero duplicates, correct verdicts, and no repair/fallback across five runs per fixture. It is a coverage-checked **union-merge** of already-structured findings plus JSON schema — not a second reviewer and not a filter: identity-tracked coverage plus a conservative source-evidence merge check restore anything the judge drops or over-merges. Judged output preserves validated lane findings up to the shared 80-finding publishing cap; if a repaired or fallback union exceeds that cap, it retains the strongest severities and reports the omitted count in the visible judge mode. If only one of several configured lanes succeeds, that survivor posts directly because there is nothing to merge. Thinking/reasoning is pinned to `minimal`. In single-job `role: all`, lane deadlines reserve a meaningful judge window; if the lanes or caller's job deadline consume that window, the posted review explicitly uses the deterministic union. Give `role: all` callers at least a 25-minute job timeout (the reusable matrix workflow uses separate jobs and is not constrained by this shared envelope).

Alternatives (do not change the default unless you mean to):

| Slug | When to use |
| --- | --- |
| `google/gemini-3.1-flash-lite` | Lower-latency legacy default; the decision benchmark found lower recall and precision |
| `anthropic/claude-haiku-4.5` | An unbenchmarked cross-vendor override |

A judge schema or transport failure is labeled on the posted review and falls back to the deterministic coverage-preserving union, subject to the same visible 80-finding publishing cap. Invalid lane artifacts or action-wide contract errors still fail closed.

## Recommended caller concurrency

This action runs **one review per invocation**. It does not implement workflow concurrency or org-specific merge gates (those belong in your reusable caller).

Use **separate first-pass and follow-up jobs**, or distinct concurrency groups, so a `synchronize` run cannot cancel an in-progress `full-pr` review.

## Agentic loops and merge gates

The initial review must finish before the fixing agent starts its review/fix/push cycle. A verify round uses the prior trusted review ledger and carries unresolved findings forward. Keep the posting identity (`bot_login`) consistent, and leave the action's hidden review metadata intact. A custom PAT or GitHub App token requires the matching posting login.

**Job success is not a clean-review gate.** `fail_on: never` posts findings without failing for them. `fail_on: bugs` and `fail_on: any` fail for matching open findings, but neither independently rejects every `partial` result. A caller requiring a clean result should inspect `verdict` as well. For the direct action example, give the action step `id: review`, then append this step in each review job:

```yaml
      - name: Require a complete clean review
        if: always()
        env:
          REVIEW_OUTCOME: ${{ steps.review.outcome }}
          REVIEW_VERDICT: ${{ steps.review.outputs.verdict }}
        run: |
          test "$REVIEW_OUTCOME" = "success" && test "$REVIEW_VERDICT" = "clean"
```

This gate rejects missing output, failure, and partial or issue-bearing reviews. It does not require every configured model to succeed: a surviving lane can produce a clean review under the action's documented failure policy. If your merge policy requires all lanes, check that separately. The reusable matrix workflow currently exposes neither review outputs nor `judge_needed` as a caller input; use the direct action when you need this output-based gate or a forced single-lane judge.

Your outer agent loop still needs to confirm that the reviewed head matches the **current** PR head, that review and required CI actually ran, and that findings have been fixed or explicitly adjudicated. Keep existing test and security gates. The two conditional jobs in the example are not a ready-made branch-protection policy; use a final required gate that accounts for which round should run and does not accept a skipped review as success.

If your coding agent creates PRs or pushes fixes from a workflow using `GITHUB_TOKEN`, those events ordinarily do not trigger another workflow. Use an appropriately scoped GitHub App/user token for the authoring operation, or design an explicit dispatch path. This README's copy-paste workflows do not include dispatch triggers. See GitHub's [workflow triggering rules](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow).

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Review jobs are skipped | The examples exclude drafts, fork PRs, and Dependabot PRs. Mark an eligible draft ready to start its initial review. These exclusions need to be accounted for in merge policy. |
| No run appears after an agent pushes | Check event filters and whether the authoring operation used `GITHUB_TOKEN`; see the trigger rules above. |
| `OPENROUTER_API_KEY` is empty / authentication fails | Check the Actions secret name, organization-secret repository access, OpenRouter key validity, credit, and model access. Do not print the secret to debug it. |
| GitHub reports permission denied / resource not accessible | Check `contents: read`, `pull-requests: write`, organization policy, and token identity. Fork and [Dependabot events](https://docs.github.com/en/code-security/reference/supply-chain-security/troubleshoot-dependabot/dependabot-on-actions) have additional token/secret restrictions. |
| Verify says no prior review exists | Finish an initial `full-pr` review first. A failed or partial initial review does not publish authoritative loop state. Check `bot_login` if the posting token changed. |
| Review is partial, or tools cannot inspect files | Read the posted diagnostic. Check full-depth checkout, reviewed commit availability, tool/diff budgets, and whether a new push made the result stale. Do not treat an empty findings list as clean. |
| Green job, but findings remain | `fail_on` defaults to `never`. Choose a finding policy and, if required, the explicit verdict gate above. |
| Cost is higher than expected | Count lanes, judge calls, follow-up rounds, and duplicate installed review workflows. Reported provider cost can be incomplete; compare it with your OpenRouter usage. |

## Inputs

| Input | Default | Notes |
| --- | --- | --- |
| `role` | `all` | `all` (collect + lanes + optional judge + post) \| `setup` (parse matrix) \| `lane` \| `judge` |
| `models` | `x-ai/grok-4.6` | Comma-separated OpenRouter slugs. Length = lane count. Cap 4. |
| `judge_model` | `openai/gpt-5.6-luna` | Independent of `models`. Used whenever judging is enabled. |
| `judge_needed` | _empty_ | `true` enables the judge (and forces it for one configured lane); `false` skips it. Empty infers from `models`. A sole survivor from a multi-lane run posts directly. |
| `github_token` | `${{ github.token }}` | Needs `pull-requests: write` to post the review. |
| `github_timeout_seconds` | `120` | Per-operation `gh` or fallback local-git timeout; 1–600. |
| `job_budget_seconds` | `1320` | Total action budget measured from Python startup. Keep this below the enclosing job timeout so publication and cleanup retain time. |
| `all_role_deadline_seconds` | _empty_ | Optional cap for concurrent lanes in `role=all`. Empty derives the cap from `job_budget_seconds` after judge and publication reserves. |
| `pr_number` | PR that triggered the workflow | Required for `workflow_dispatch`. |
| `fail_on` | `never` | Finding policy: `never` \| `bugs` \| `any`. Operational/schema errors always fail. |
| `roast_level` | `professional` | `professional` \| `playful`. |
| `custom_instructions` | _empty_ | Extra prompt text, max 16,000 UTF-8 bytes. Never put secrets here. |
| `path_profiles` | _empty_ | Caller-owned additive review profiles: JSON `[{name?, paths: [globs], instructions}]`, applied only when a changed path matches (`*`/`?` stay within a path segment, `**` crosses). Sharpen attention; never narrow the review. Trusted workflow config only — never interpolate PR content, never put secrets here. Max 20 profiles and 16,000 UTF-8 bytes. |
| `status_comments` | `true` | Live status comment on the PR. |
| `max_diff_kb` | `300` | Embedded diff cap. Over-budget diffs go through **diff-budget triage**: generated/vendored/lock-class files (the reviewed commit's `.gitattributes` `linguist-generated`/`linguist-vendored`, lockfile heuristics, large committed JSON snapshots, `generated_paths`) demote to stubs first, then the largest hand-written files, so hand-written hunks keep the budget. A stubbed file stays in the embedded diff (header + counts + first-hunk reference), is materialized into the inert checkout for the tools even past the normal 1 MB cap (up to 8 MB), and still requires a coverage entry — so a fully stubbed-or-embedded diff keeps its real verdict and review-loop continuity. `.gitattributes` is repository content (PR-author-controlled); honoring it only shifts packing priority — a demoted file keeps its stub, coverage obligation, and tool access, which is strictly safer than the raw byte cut it replaces (where tail files vanished entirely). Files dropped entirely, an unparseable diff's raw byte cut, or stubs with tools disabled (`max_tool_turns: 0`) ⇒ `partial`, never clean. |
| `generated_paths` | _empty_ | Extra globs (JSON array of strings) classified as generated/vendored during diff-budget triage. Demotion only shifts packing priority — never excludes a file from review. Trusted workflow config only — never interpolate PR content. Max 8,000 UTF-8 bytes, 200 globs. |
| `review_scope` | `full-pr` | `full-pr` \| `latest-commit`. Initial rounds require `full-pr`. |
| `review_mode` | `auto` | `auto` (opened = initial, synchronize = verify) \| `initial` \| `verify`. |
| `effort` | _empty_ | Optional OpenRouter reasoning effort for **review lanes**. |
| `max_tool_turns` | `50` | Read-only tool rounds against the inert checkout. `0` disables tools. First-pass default matches the sibling Grok `max_turns`. Follow-up jobs may pass `30`. |
| `openrouter_timeout_seconds` | `180` | Per-request OpenRouter timeout; 1–600 seconds. |
| `lane_index` | `0` | Zero-based matrix index used by `role=lane` artifact naming. Normally supplied by the reusable workflow. |
| `lane_model` | _empty_ | Optional validated model override for `role=lane`. Normally supplied through matrix plumbing. |
| `lane_results_dir` | _empty_ | Lane artifact output/input directory used by `role=lane` and `role=judge`. Normally supplied by orchestration. |
| `head_sha` | _empty_ | Full reviewed commit SHA resolved by reusable-workflow setup. Internal lane/judge plumbing; pull-request runs use the event head by default. |
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
| `judge_needed` | Whether the judge is enabled for the configured lanes |
| `judge_model` | Judge slug used when judging is enabled |
| `lane_file` | Written lane JSON (`role=lane`) |
| `lane_ok` | `true` when `role=lane` produced a valid structured artifact |

## Source evidence and incomplete output

The judge proposes grouping; published text, severity, attribution and anchors are
rebuilt from validated lane findings. Uncertain matches stay separate, which can
produce more duplicate-looking comments than free-form semantic merging. Judge
rewrites cannot downgrade a source bug or substitute unsupported evidence.

Lane overflow retains the strongest severities, records `dropped_findings` in the
artifact, and produces a visible partial review without an authoritative ledger.
Malformed candidates anywhere in the response still fail validation. The separate
union publishing cap remains visible in the judge mode. Reported judge charges
are included even when its answer is unusable; unknown attempted charges are
labeled incomplete.

## Local tests

CI on this repo does not need a live `OPENROUTER_API_KEY`.

For measuring review recall against fixtures with known defects (which DOES
spend tokens and needs `OPENROUTER_API_KEY`), see the offline bench in
[bench/README.md](bench/README.md).

```bash
python3 -m pip install -e '.[dev]'
ruff check src tests bench
pytest
```

## License

[MIT](LICENSE). Copyright (c) 2026 Fly Over Coder.
