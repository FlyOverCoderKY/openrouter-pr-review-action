# Changelog

## Unreleased

Loop-protocol hardening toward Grok parity — first changesets of the merged parity plan in `docs/pr-review-harness-independent-review.md`.

- Preserve `reasoning_details` (or normalized `reasoning`) on assistant messages echoed back during tool turns, per OpenRouter's reasoning contract. Previously every turn dropped the model's prior reasoning from context.
- Withdraw tools *before* the request that follows the final permitted tool round. The old loop kept offering tools and then answered an over-budget round with an assistant `tool_calls` entry and no tool results — an invalid conversation that OpenAI-compatible providers reject, failing the lane after the entire budget was spent. If a model still emits tool calls after withdrawal, they now get stub results (bounded by a repair cap) instead of a protocol violation.
- One schema-enforced, tool-free finalization retry when a tool-backed run's natural finish is not valid findings JSON. The tool path runs without `response_format`, so a malformed finish previously failed the lane open immediately.
- Retry transient OpenRouter errors (408/429/5xx, timeouts, connection failures) with exponential backoff, honoring `Retry-After`. Previously a single transient failure anywhere in a 50-round loop discarded all accumulated tool work.
- Salvage finalization: a mid-loop failure that survives the retries now asks the model once — tools withdrawn, schema attached — for findings from the evidence already gathered, instead of failing the lane. A context-overflow failure additionally truncates old tool observations (keeping the newest two intact) so the salvage request fits.
- Lane observability: lane artifacts and the posted review now carry `requests`, `tool_rounds`, `retries`, `cached_tokens` (OpenRouter prompt-cache telemetry), and a `salvaged` flag. Failed lanes report their token usage too.

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
