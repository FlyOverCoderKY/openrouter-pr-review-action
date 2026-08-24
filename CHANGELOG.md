# Changelog

## 0.1.0 — 2026-08-24

Initial public v1.

- OpenRouter-only PR review action with a custom chat-completions harness and read-only tools (`read_file`, `grep`, `list_dir`) against an inert checkout of the reviewed commit.
- Default review lane: `x-ai/grok-4.6` (verified live on OpenRouter).
- `models` is a comma-separated slug list; length is the lane count; hard cap of 4.
- Judge is **self-disabling** when only one review lane is configured: that lane is posted directly with no OpenRouter judge call and no merge/de-dupe.
- Two or more lanes require `judge_model` (default `google/gemini-3.1-flash-lite`, verified live). The judge merges/de-dupes structured findings with JSON schema and `reasoning.effort=minimal`. Schema mismatch fails closed.
- Documented judge alternatives: `openai/gpt-4.1-nano` (cheaper/faster) and `anthropic/claude-haiku-4.5` (stricter merge). Not defaults.
- Caller-facing review product: `review_scope` (`full-pr` / `latest-commit`, no silent full-PR fallback), `review_mode` (`auto` / `initial` / `verify`), `fail_on`, `max_diff_kb` (truncated diffs are `partial` and never clean), `status_comments`, professional default tone.
- Reusable workflow for setup → parallel lane jobs → judge. GitHub bills those minutes in parallel.
- Reserved unused `persona` hook. A future single-persona run should skip the judge (one reviewer = no judge). Personas are not implemented in v1.
