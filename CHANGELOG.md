# Changelog

## Unreleased

Loop-protocol hardening toward Grok parity — first changesets of the merged parity plan in `docs/pr-review-harness-independent-review.md`.

- Preserve `reasoning_details` (or normalized `reasoning`) on assistant messages echoed back during tool turns, per OpenRouter's reasoning contract. Previously every turn dropped the model's prior reasoning from context.
- Withdraw tools *before* the request that follows the final permitted tool round. The old loop kept offering tools and then answered an over-budget round with an assistant `tool_calls` entry and no tool results — an invalid conversation that OpenAI-compatible providers reject, failing the lane after the entire budget was spent. If a model still emits tool calls after withdrawal, they now get stub results (bounded by a repair cap) instead of a protocol violation.
- One schema-enforced, tool-free finalization retry when a tool-backed run's natural finish is not valid findings JSON. The tool path runs without `response_format`, so a malformed finish previously failed the lane open immediately.
- Retry transient OpenRouter errors (408/429/5xx, timeouts, connection failures) with exponential backoff, honoring `Retry-After`. Previously a single transient failure anywhere in a 50-round loop discarded all accumulated tool work.
- Salvage finalization: a mid-loop failure that survives the retries now asks the model once — tools withdrawn, schema attached — for findings from the evidence already gathered, instead of failing the lane. A context-overflow failure additionally truncates old tool observations (keeping the newest two intact) so the salvage request fits.
- Lane observability: lane artifacts and the posted review now carry `requests`, `tool_rounds`, `retries`, `cached_tokens` (OpenRouter prompt-cache telemetry), and a `salvaged` flag. Failed lanes report their token usage too.
- Transcript bounding: `read_file` gains ranged reads (`start_line`, `max_lines`) and the unranged per-read cap drops from 200 KB to 60 KB with a continuation hint (every observation is resent on every later request, so single large reads multiplied across the whole run). A per-lane aggregate observation budget (600 KB) withdraws tools the same way the turn budget does. Checkpointed compaction is deferred until cached-token telemetry from live runs says what it is worth.
- Complete latest-commit diffs: `compare_diff` and `commit_diff` now fetch the raw `application/vnd.github.diff` media type instead of rebuilding a diff from the REST JSON `files` array (which caps at 300 files with no truncation marker and omits `patch` for large files). `compare_diff` also validates the range is a linear fast-forward first; a force-pushed range now falls back to the single latest commit with a visible notice instead of silently reviewing a merge-base diff.
- Workspace preparation fails closed: if the inert checkout cannot be materialized the run errors (with a pointer to `max_tool_turns: 0` for a deliberate tool-less review) instead of silently posting an unmarked, tool-less review.
- SHA-bound artifacts and publication: lane artifacts record the reviewed `head_sha`; the judge refuses to merge lanes that reviewed different commits (fail-closed); the posted review and its `commit_id` pin to the commit the lanes actually reviewed; and if the PR head advanced past it, the review is marked `partial` with a visible notice. A latest-commit fallback diff is now also `partial`, never `clean`.
- `role=lane` no longer collects the PR twice when it posts directly (single-lane path).
- Output hardening (parity with the sibling Grok action): model-authored finding and judge paths are validated as safe repository-relative paths (traversal, backticks, backslashes, control characters, and absolute or oversized paths are dropped to `null`, keeping the finding); `@`-mentions in model-authored titles, bodies, and lane errors are neutralized with a zero-width space so a posted review can never ping users or teams; and long findings lists continue in follow-up comments (`Part N of M`) instead of being truncated out of the review body.
- Coverage manifest (the sibling Grok action's real anti-glance mechanism): initial reviews must return a `coverage` array accounting for every embedded-diff file, including zero-finding files. The schema-enforced finalization requires it; a manifest that misses a diff file fails the lane open with a clear error; per-file counts that disagree with the reported findings post a visible note while keeping the completed verdict (Grok 1.0.5 semantics). Lane artifacts carry `coverage` and (for verify rounds) `resolutions`.
- **Review loop**: convergent review → fix → re-review rounds, ported from the sibling Grok action with the improvements the harness review called for.
  - A hidden **ledger** (base64 marker in the bot's own review bodies, bound to repository, PR, reviewed SHA, and ledger version) carries open findings across rounds — each with a bounded **evidence** excerpt, not just a clipped title. Only reviews authored by the new `bot_login` input (default `github-actions[bot]`) are trusted. State recovery fails closed: a corrupted newest marker raises with a reset instruction instead of silently restarting the loop.
  - **Verify rounds** (synchronize / `review_mode: verify`) list the carried findings with evidence in the prompt, require a `resolutions` entry per finding (`fixed` / `not_fixed` / `fixed_incorrectly` / `disputed`; a reasoned rebuttal settles a dispute), post a visible per-round resolution report (✅/❌/⚠️/🤝/⏳), and carry unresolved findings forward. Multi-lane resolutions merge deterministically and conservatively (a finding is fixed only if every lane says fixed; any dispute settles it) — no LLM judge involvement.
  - **Reviewed-SHA continuity**: verify rounds diff from the last successfully published ledger SHA, not just the webhook `before`, so a follow-up that raced a cancelled run can never skip the commits that run was reviewing. A state-free synchronize under `latest-commit` scope is refused with instructions to run an initial full-PR review (fail-closed, matching the sibling).
  - **Inline comments**: findings that anchor to a diff file post as inline review comments with a hidden `<!-- or-finding:ID -->` marker (out-of-diff blast-radius findings stay body-only; inline failures fall back to a body-only review). Replies to those threads and recent PR comments are harvested into the next verify round's prompt as untrusted fixing-agent context.
  - `issue_count` / `bug_count` outputs now report **open** findings after the round (carried plus new), `fail_on` applies to them, and a new `round` output reports the loop position. A partial or error run never publishes ledger state, so the previous marker remains the retry boundary.

## 1.1.0 — 2026-08-25

Review-depth parity with the sibling Grok action's first-pass budget. Still OpenRouter-only (`OPENROUTER_API_KEY` → openrouter.ai). No Grok CLI, no `XAI_API_KEY`, no `api.x.ai`.

- Default `max_tool_turns` is **50** (was 8), matching the sibling first-pass `max_turns` default. Still configurable; `0` disables tools. Follow-up jobs may pass a lower value (sibling callers often use 30). The reusable workflow now exposes the input.
- Review prompt requires read-only tool use for **blast radius**: filename-inventory tests, README / code-map docs, and sibling CI/workflow files — not just the embedded diff. Changed paths are listed in the user prompt. Out-of-diff findings are valid. A workflow-only PR that is missing from README or a code-map inventory test should not glance clean.
- Harness no longer attaches JSON-schema `response_format` while tools are offered, and nudges once (with `tool_choice=required`) if the model tries to finish with zero tool calls. That glance-and-clean path is what missed a workflow-inventory test on a 31-line YAML PR.
- `fail_on` remains `never`. One-lane judge skip unchanged.

## 1.0.0 — 2026-08-24

Initial public v1.

- One-lane Grok 4.6 live smoke on PR #2 posted a real `issues` review with the judge skipped.
- OpenRouter-only PR review action with a custom chat-completions harness and read-only tools (`read_file`, `grep`, `list_dir`) against an inert checkout of the reviewed commit.
- Default review lane: `x-ai/grok-4.6` (verified live on OpenRouter).
- `models` is a comma-separated slug list; length is the lane count; hard cap of 4.
- Judge is **self-disabling** when only one review lane is configured: that lane is posted directly with no OpenRouter judge call and no merge/de-dupe.
- Two or more lanes require `judge_model` (default `google/gemini-3.1-flash-lite`, verified live). The judge merges/de-dupes structured findings with JSON schema and `reasoning.effort=minimal`. Schema mismatch fails closed.
- Documented judge alternatives: `openai/gpt-4.1-nano` (cheaper/faster) and `anthropic/claude-haiku-4.5` (stricter merge). Not defaults.
- Caller-facing review product: `review_scope` (`full-pr` / `latest-commit`, no silent full-PR fallback), `review_mode` (`auto` / `initial` / `verify`), `fail_on`, `max_diff_kb` (truncated diffs are `partial` and never clean), `status_comments`, professional default tone.
- Reusable workflow for setup → parallel lane jobs → judge. GitHub bills those minutes in parallel.
- Reserved unused `persona` hook. A future single-persona run should skip the judge (one reviewer = no judge). Personas are not implemented in v1.
