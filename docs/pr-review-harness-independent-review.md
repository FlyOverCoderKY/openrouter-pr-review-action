# Independent review of the Grok and OpenRouter PR-review harnesses

Original review: 2026-08-25, Codex GPT-5.6 Sol.
Updated: 2026-08-25 — merged with a second independent review (Claude Fable 5, same prompt, no shared context), reconciliation of the two, Nathan's program decisions, and a live verification of the org merge-gate state.
Revised again 2026-08-25 after Codex's second-round feedback: gate-authorization boundary made explicit, compaction reframed as checkpointed, GitHub pagination constants settled empirically, exhaustion remediation improved, and several claims recalibrated.

Finding attribution: `[Codex]`, `[Claude]`, or `[both]`. Where the two reviews disagreed, the disagreement and its resolution are stated rather than papered over.

## Scope and baseline

This document compares the current production revisions of the two review harnesses and the RetireGolden reusable workflows that invoke them:

- Grok harness `v1.0.5`: `3989e7854808f4d9f48e4742a1419148d9aab3fc`
- OpenRouter harness `v1.1.0`: `2d338f42321747984e0088ca42ce635dcf605242`
- RetireGolden org reusable containing the OpenRouter pin: `54bde42f4bb9e14b6c3c7e3126c5d582abfea4b9`
- RetireGolden org reusable containing the Grok pin: `fe20f331e17eb390f4b992b6dfbe3e46ddfc227f`

At the time of review, each harness's `main` branch matched the cited release commit, the org `main` matched `54bde42f`, and the named OpenRouter callers pinned the org reusable at the full `54bde42f` SHA. The floating `v1` aliases are stale (Grok `v1` → v1.0.3, OpenRouter `v1` → v1.0.0) but are not part of the reviewed invocation path.

The older RetireGolden #311 and RetireGolden-Pro #210 OpenRouter runs used `v1.0.0` (8 tool turns) and are not evidence for or against `v1.1.0` parity. As of this document there is still no true v1.1.0 bake-off data.

## Program decisions (Nathan, 2026-08-25)

These decisions change how the findings below should be read.

1. **The Grok action is frozen.** No changes land in `grok-pr-review-action`, ever — not even for its own defects. Its current observed behavior is the parity bar. Grok-side findings are retained below as reference and as *do-not-copy* patterns only.
2. **Goal:** bring the OpenRouter harness to parity or better, run a few days of side-by-side testing (OpenRouter non-required throughout), then cut over to OpenRouter only and retire the Grok lane — thin callers, org reusable, and eventually the `XAI_API_KEY` secret.
3. **Follow-ups:** port a Grok-style ledger (carried findings, per-finding resolutions, reviewed-SHA continuity) — but with bounded evidence per finding, not 80-char titles (see the lossy-ledger finding).
4. **Inline comments:** port per-finding inline comments with hidden markers and reply harvesting for dispute handling.
5. **Motivation is all of:** model flexibility, reliability/simplicity (no closed-source CLI, no bwrap/AppArmor machinery), and cost/billing consolidation — plus the explicit destination beyond parity: multi-model bake-offs, agent personas, and a results judge on the custom harness. Consequence: SHA-binding lane artifacts and judge hardening are planned pre-multi-lane work, not latent concerns.
6. **Merge gate (verified live, 2026-08-25):** contrary to prior recollection, **no OpenRouter gate exists anywhere yet.** Required checks on `main` by ruleset:

   | Repo | Required checks |
   | --- | --- |
   | RetireGolden | lint, test, e2e, build, `Scan (p/default)`, `ZAP DAST / ZAP Baseline`, CLA, `review / grok-first-pass-gate` |
   | RetireGolden-MCP | test, `review / grok-first-pass-gate` |
   | retiregolden.org | `review / grok-first-pass-gate` |
   | RetireGolden-Pro | `review / grok-first-pass-gate` |
   | RetireBench | *(none — no required checks at all; worth a separate decision)* |

   The OpenRouter org reusable at `54bde42f` deliberately has no gate (its header forbids one). Plan: build an OpenRouter first-pass marker + gate job in the org reusable (surfacing as `openrouter / <gate-job>` under the current caller job ids), bound to the **reviewed head** rather than Grok's carry-forward semantics, and run it **advisory (non-required)** throughout. Any ruleset change — requiring the OpenRouter gate, removing the Grok entries — is a separate, explicitly Nathan-approved manual operation at cutover time. No implementation changeset may edit rulesets or required checks.

## Executive summary

Both reviews independently reached the same architecture story: the Grok action delegates the agent loop to the native pinned Grok CLI, while the OpenRouter action implements a stateless Chat Completions loop that resends the full accumulated conversation — original diff plus every prior tool observation — on every request, so later calls become larger, slower, and more attention-diluting. Both reviews found **no P0 defects**.

The reviews diverged on the *first* fix. Codex ranked transcript compaction as the leading parity improvement. Claude ranked three cheaper loop fixes ahead of it: the harness discards the model's reasoning between turns (a provider-contract violation that removes prior reasoning from every turn's context), has zero retry so a single transient error discards all accumulated work, and its budget-exhaustion path builds a conversation the provider will likely reject. Resolution (see the merged plan): the loop fixes are small, well-bounded changes and land first; a *cheap* transcript bound (harder read caps, ranged reads) lands with them; full compaction follows and must be **checkpointed** — editing earlier messages in place necessarily invalidates the cached prefix, so compaction means paying one deliberate cache miss to write a summary epoch, then staying append-only until the next threshold.

Matching nominal turn limits, forcing one arbitrary tool call, adding lanes or a judge, or raising Grok's `max_turns` cannot address the structural differences. (Grok stays frozen regardless.)

## How the loops differ

| Concern | Grok `v1.0.5` | OpenRouter `v1.1.0` |
| --- | --- | --- |
| Execution | Native Grok CLI manages the agent and tool loop. | A custom Python loop makes repeated OpenRouter Chat Completions requests. |
| Tools | CLI-provided `read_file`, `grep`, `list_dir`. | Python implementations of the same trio, dispatched in-process. |
| Sandbox | Strict bubblewrap sandbox; shell, web, writes, memory, subagents disabled. Agent-control files and symlinks excluded from the snapshot. | No model-accessible shell or arbitrary code execution: a fixed read-only dispatcher over an inert `git archive` checkout (the harness itself performs fixed git/filesystem/regex operations). No shell, write, or network tool is exposed to the model. Secret-like paths refused. Structurally the stronger model. |
| Reasoning continuity | Maintained by the CLI within one session. | Discarded: `reasoning`/`reasoning_details` are dropped from every echoed assistant turn. |
| Transient errors | CLI-internal retry behavior. | None: one failed HTTP call anywhere in the loop fails the lane and discards all work. |
| First-pass prompt | Exhaustive multi-sweep review, every severity, plus an **enforced per-file coverage manifest**. | Shorter prompt with a mandatory, heavily CI/docs-oriented blast-radius pass; no coverage manifest. |
| Follow-up prompt | Prior finding IDs, locations, severities, agent replies, severity floors, escalation state. | `verify` only changes prompt wording; no prior findings or resolution state supplied. |
| Budget | `max_turns=50` limits native CLI agent turns. | `max_tool_turns=50` counts assistant responses containing tool calls; nudge/final calls are additional and one response may carry multiple tool calls. The two limits are not equivalent units. |
| Structured result | Requires CLI `EndTurn`, then parses summary, issues, resolutions, coverage; strict validation. | `response_format` json_schema is attached only when tools are absent or exhausted; the normal tool-backed final text is parser-validated, not provider-schema-enforced. |
| Incomplete verdicts | Truncation, single-commit fallback, and stale head all marked `partial`. | Only diff truncation is `partial`; single-commit fallback or a stale head can still produce `clean`. |
| Judge | None (optional cheaper `verify_model` tier instead). | Schema-constrained clerical merge judge for ≥2 lanes; the org runs one lane, so it is idle today. |
| Checkout/SHA | Materializes the exact collected commit; rechecks the live PR head before publishing; withholds ledger when stale. | Materializes the collected commit, but materialization failure silently disables tools; publication does not recheck the live head; lane artifacts carry no SHA. |
| Follow-up continuity | Starts from the last successfully published ledger SHA; carries unresolved findings. | Uses the current webhook `before...after` range; no persistent state. |
| Output hardening | Path validation, `@`-mention neutralization, multi-part bodies. | None of the three; body hard-truncates at 60k chars. |

## Findings — OpenRouter harness (the work queue)

### P1 [both] — Tool transcripts grow without bound

`src/or_pr_review/harness.py::_run_loop`, `_run_one_tool`; `src/or_pr_review/workspace.py::tool_read_file`

Every assistant message and complete tool observation is appended to `conversation` and resent on every request. `read_file` returns up to **200 KB per call** (the 40 KB `_cap_output` limit applies only to `grep`), multiple tools can fire in one round, and the org reusable embeds diffs up to 1 MB. Final context grows monotonically; cumulative transmitted context grows roughly quadratically with rounds. Provider prompt caching may blunt billing but not context-length pressure or attention dilution.

Fix: a cheap bound now (harder per-read caps, ranged file reads, smaller aggregate observation budget); **checkpointed compaction** later — any edit to an earlier message invalidates the cached prefix, so compaction is a deliberate epoch: pay one cache miss to fold old observations into a summary, then stay append-only until the next threshold. Before any cache-driven optimization, pin sticky provider routing and read OpenRouter's cached-token usage telemetry so effects are measured rather than assumed.

### P1 [Claude] — Budget-exhaustion path builds an invalid conversation

`src/or_pr_review/harness.py::_run_loop` (exhaustion branch)

When `turns > max_tool_turns`, the loop appends the assistant message *with its `tool_calls`* and then a `user` "finish now" message — never the `tool` results for those call ids. OpenAI-compatible validators (including xAI, the default lane) reject assistant tool_calls not followed by matching tool messages, so the finish request is expected to 400 → `LaneError` → the entire lane fails open after burning the full budget. It fires precisely on the runs that worked hardest. No test covers the branch. Fix: never solicit a tool call the harness cannot service — after executing the final permitted tool round, the next request must omit `tools` (and attach the schema) so an unserviceable round cannot arise; synthetic tool-result stubs remain only as a repair if a dangling round still occurs.

### P1 [Claude] — Zero retry/backoff and no salvage path

`src/or_pr_review/harness.py::openrouter_chat`

A single transient 429/5xx/timeout anywhere in a worst-case loop of 52+ requests (50 executed tool rounds, the over-budget round, the finish call, plus possible nudge and schema-retry calls) discards all accumulated tool work (`tests/test_harness.py` codifies this as expected). With growing payloads and a 180 s per-request timeout, the latest — most valuable — turns are the most likely to die. There is also no "return findings from what you have gathered" fallback on failure or context overflow. Fix: bounded retry with backoff, plus a salvage finalization call.

### P1 [Claude] — Reasoning continuity discarded between turns

`src/or_pr_review/harness.py::_assistant_record`

Only `content` and `tool_calls` are echoed back; `reasoning`/`reasoning_details` are dropped. OpenRouter's reasoning-token guidance requires `reasoning_details` to be returned unmodified during tool calling, so this is a provider-contract violation outright, and a likely contributor to "later turns get slower and thinner": prior reasoning is absent from every turn's context. The code change is small; the size of the impact is an empirical question for the test window, not an assumption.

### P1 [Codex] — Latest-commit collection can be silently incomplete

`src/or_pr_review/github_ops.py::compare_diff`, `commit_diff`, `_patches_from_files`

The latest-commit path reconstructs a unified diff from the GitHub REST JSON `files` array with no pagination handling and no completeness marker. Constants settled empirically (2026-08-25): the compare endpoint hard-caps the files array at 300 with no truncation marker the harness reads (a ~14,000-file kernel range returned exactly 300 files); the commit endpoint returns the complete files array for commits up to 300 files regardless of `per_page` (35- and 66-file commits came back whole with no parameters — there is no 30-file default page), then paginates via Link headers up to 3,000 files, which an unpaginated `gh api` call never follows. `patch` may also be absent for large files. Missing files never reach the verdict. The compare path also never validates that the range is a linear fast-forward (`[both]` — Grok requires `status=="ahead"`/`behind_by==0` and falls back to single-commit with a notice; after a force-push OpenRouter silently reviews a merge-base-relative diff). Fix: use the raw `application/vnd.github.diff` media type as Grok does, validate compare status, and mark any incomplete collection `partial`.

### P1 [both] — Verification mode is a delta review, not verification

`src/or_pr_review/collect.py::resolve_mode`; `src/or_pr_review/prompt.py::_system_prompt`; org `openrouter-code-review.yml::openrouter-follow-up`

`verify` changes prompt wording only: no prior findings, no resolutions, no carried state, no last-reviewed SHA. The org workflow cancels older follow-ups (`cancel-in-progress: true`), and a later webhook range cannot recover the commits a cancelled run was reviewing — Grok closes exactly this hole via `before = ledger.reviewed_sha`. Single-commit fallback can still yield `clean`. Resolution: the **ledger port** (program decision 3) is the fix; until it lands, the org reusable should serialize follow-ups instead of cancelling.

### P1 [both] — Tool availability fails open; schema never returns on the happy path

`src/or_pr_review/cli.py::_prepare_workspace`; `src/or_pr_review/harness.py::_run_loop`

If the inert checkout cannot be materialized, the run logs a warning and continues with `workspace=None`: no tools, schema attached from turn one, a system prompt that still mandates tool use, and a posted review with no marker that anything degraded — a silent toolless `clean` is exactly the glance v1.1.0 was built to kill. Separately `[Codex]`: on the normal tool-backed path, the JSON schema is withheld while tools are offered and is *never re-attached* when the model stops calling tools naturally — only after budget exhaustion — so the advertised schema-enforced finish is absent from the common success path. Fix: fail closed (or visibly mark toolless reviews), and add a distinct schema-enforced, tool-free finalization call.

### P1 [Codex] / P2-latent [Claude] — Lane and judge artifacts are not bound to a commit

`src/or_pr_review/schema.py::LaneResult`; `src/or_pr_review/cli.py::_role_lane`, `_role_judge`, `_finish`

Lane artifacts carry findings and usage but no reviewed SHA or diff identity. In the multi-job path each lane collects the *live* head independently and the judge re-collects and publishes against whatever head exists then — lanes can review different commits and the judge can stamp a third. The production single-process `role=all` path avoids the cross-job variant (hence Claude's lower severity today) but still never rechecks the live head before publishing. **Given the multi-lane/bake-off destination (program decision 5), this is a hard precondition, not a latent nit.** Fix: reviewed SHA + canonical diff hash in every artifact, one immutable collection manifest through setup/lanes/judge, reject mixed artifacts, recheck head before publication, mark stale output `partial`.

### P2 [both] — Model-authored review fields published without output hardening

`src/or_pr_review/schema.py::parse_finding`; `src/or_pr_review/merge.py::format_issue_block`; `src/or_pr_review/publish.py::render_review`

Finding paths are not validated as safe repo-relative paths; titles/bodies/locations are emitted raw into Markdown; `@mentions` are not neutralized (a hostile diff can induce the model to ping `@org/team` in a posted review); the body hard-truncates at 60k chars, dropping findings, where Grok chunks into continuation comments. Fix: port Grok's `_valid_review_path` + `neutralize_mentions` + multi-part rendering.

### P2 [Claude] — Model-supplied regex runs with no timeout

`src/or_pr_review/workspace.py::tool_grep`

A model-authored pattern is compiled with Python's backtracking `re` and run over up to 5,000 files; a catastrophic pattern can hang the lane until the 30-minute job timeout, which posts nothing (see org findings). Self-DoS/cost only. Fix: per-call time budget (or a bounded regex subset).

### P2 [Claude] — Fence breakout in the prompt

`src/or_pr_review/prompt.py::_fence`

Untrusted PR title/body go into ```` ```text ```` fences with backticks unescaped; a body line containing a bare fence closes the block and can forge section headers (e.g. a fake "Caller instructions" section). Diff content is prefix-protected; title/body are not. Defense-in-depth fix: escape or pad fences.

### P2 [Claude] — Stale `v1` alias points at the pre-parity build

Repo tags: `v1` → `034e7869` = v1.0.0 (8-turn glance build). Org and callers are SHA-pinned and unaffected, but any consumer following `@v1` gets pre-parity behavior. Fix: advance the alias after a verified release (deleting a published tag can break consumers and is not an equivalent option).

## Findings — org reusable (`RetireGolden/.github`)

### P2 [Claude] — `max_diff_kb: 1024` is Grok-tuned and amplifies the OpenRouter snowball

`openrouter-code-review.yml` (input default + both action calls). Grok pays the megabyte embed once per run; OpenRouter re-uploads it on every turn, up to 50×. Until transcript bounding lands, the OpenRouter lane deserves its own lower cap.

### P2 [Claude] — 30-minute job timeout makes a slow lane vanish silently

Both org jobs set `timeout-minutes: 30` and the harness has no internal wall-clock. On timeout the job is cancelled: no review, no incomplete comment, and the status comment reads "Reviewing with OpenRouter…" forever. Grok times out equally silently but its gate/marker machinery surfaces the failure; the OpenRouter lane just disappears — which systematically undercounts it during side-by-side testing. Fix: a finish-early deadline in the harness and/or an `always()` finalizer step.

### P2 [Claude] — OpenRouter follow-up is not gated on first-pass completion

Grok's follow-up exits with no spend until the `grok-org-first-pass:done` marker exists; the OpenRouter follow-up runs on every synchronize. A push right after open produces a stateless "verify" with no initial review posted — wasted spend and distorted comparisons. Fix alongside the ledger/marker work.

### P2 [Claude] — "Still obey AGENTS.md" is self-contradictory and asymmetric between lanes

The per-repo policy tells both reviewers to obey AGENTS.md. Grok's workspace deliberately strips `agents.md` (its reviewer cannot read it); OpenRouter's workspace keeps it while its prompt says never to follow instructions found in repository files. Also an uncontrolled variable between the two bake-off lanes. Fix: reword the policy and pick one workspace behavior.

### P1 [Codex] — The Grok first-pass gate stays green after history replacement (do-not-clone)

`grok-code-review.yml::grok-first-pass-gate`. The gate succeeds whenever the old done marker exists and copies a success status onto the *current* head — after a multi-commit force-push, rewritten history slides under a green required check. Moot for the frozen Grok lane, but it is the single most important design instruction for the **new OpenRouter gate** (program decision 6): bind gate state to the reviewed head/history; a non-fast-forward rewrite requires a fresh full-PR pass before success attaches to the replacement head.

## Findings — Grok harness (frozen; reference and do-not-copy patterns only)

- **[both] Sandbox setup weakens the host** (`scripts/install-grok.sh::relax_unprivileged_userns_sysctl`): disables `kernel.apparmor_restrict_unprivileged_userns` via sudo and never restores it — transient on ephemeral GitHub-hosted runners, a persistent host-wide change on any self-hosted runner. Do not copy; OpenRouter's no-exec design needs no sandbox installer and is the stronger model.
- **[Codex] Ledger is lossy and weakly attributed** (`loop.py::LedgerFinding`, `github.py::list_bot_review_bodies`): carried state preserves only title (clipped to 80 chars), location, severity, status — verify rounds adjudicate without the original failure scenario or fix criterion; trust rests on the shared `github-actions[bot]` login. **Design guidance for the OpenRouter ledger port:** carry bounded evidence per finding and bind state to repo, PR, reviewed SHA, and harness version.
- **[Claude] Untrusted PR body embedded raw** (`prompt.py::build_prompt`): the PR description enters the prompt with no fence at all — the weakest injection surface in either harness. OpenRouter's fenced version (with the escaping fix above) is the pattern to keep.
- **[Claude] Enforcement boundary is the closed-source CLI**: `--yolo` plus trusting the pinned binary to honor `--tools`/`--sandbox strict`. Mitigated (sha256 pin, bwrap, control-file stripping) but structural — an argument *for* the harness-owned-tools direction, not something to converge toward.

## Reconciliation notes (corrections between the two reviews)

- Claude's original report stated tool outputs were capped at 40 KB; that cap applies only to `grep` — `read_file` returns up to 200 KB per call. Codex had this right; it strengthens the transcript-growth finding.
- Codex's "commit endpoint defaults to a 30-file page" constant was disputed in the first update and re-asserted in second-round feedback; empirical probes settled it (see the finding): 35- and 66-file commits return complete file arrays with no parameters, so there is no 30-file default page — files paginate via Link headers at 300, and compare hard-caps at 300. The finding's conclusion (silent incompleteness) was always valid.
- Codex's compaction suggestion is adopted in **checkpointed** form: summary epochs that accept one deliberate cache miss, append-only between epochs. (This update's earlier "prefix-stable in-place replacement" phrasing was self-contradictory — any in-place edit invalidates the cached prefix — and was corrected on Codex's second-round feedback.)
- Codex did not cover: the exhaustion-path 400, retry/salvage, reasoning continuity, grep timeouts, the 30-minute-timeout vanish, follow-up spend gating, prompt fencing, or the stale aliases. Claude did not originally cover: REST-JSON diff incompleteness, SHA-unbound artifacts, the gate carry-forward hole, or the lossy-ledger design note. The union is this document.
- Several Claude P2s (fail-open workspace, follow-up continuity) were promoted to P1 under the cutover goal: once OpenRouter is the only reviewer, its follow-ups and degradation paths are load-bearing.
- Codex's second-round feedback also produced: the explicit gate-authorization boundary (ruleset edits are Nathan-only operations), the cleaner exhaustion remediation (withdraw tools once the budget is spent rather than repairing with stubs), the 52+-request worst-case count, the "no model-accessible execution" wording, move-don't-delete tag guidance, exact ruleset context strings, and the recalibrated reasoning-continuity claim (contract violation and likely contributor; impact to be measured, not assumed).

## Merged parity plan

Implementation lands as bounded changesets, roughly: (1) loop protocol + reasoning continuity; (2) retry + observability; (3) context bounding/compaction; (4) collection + SHA correctness; (5) ledger + inline comments. Org-workflow changes stay in their own changesets, and nothing in this plan authorizes editing rulesets or required checks.

**Phase 1 — make the lane trustworthy (before the test window means anything):**
1. Echo `reasoning_details` in `_assistant_record`.
2. Retry/backoff in `openrouter_chat` plus a salvage finalization on failure or context overflow.
3. Fix the budget-exhaustion 400 (tool-result stubs before the finish request).
4. Raw-diff media type + fast-forward validation + partial marking on the latest-commit path.
5. Fail closed (or visibly mark) when the inert workspace is unavailable.
6. Cheap transcript bounding now (harder read caps, ranged reads); prefix-stable compaction later.
7. Distinct schema-enforced, tool-free finalization call once tools go quiet.

**Phase 2 — loop parity:**
8. Ledger port: carried findings with bounded evidence, per-finding resolutions, reviewed-SHA continuity, SHA/version-bound state (per the do-not-copy notes above).
9. Inline comments with hidden per-finding markers + reply harvesting for disputes.
10. Port Grok's per-file coverage manifest and validation — the actual exhaustiveness mechanism.
11. Output hardening: path validation, mention neutralization, multi-part bodies.
12. SHA-bind lane artifacts and the judge input path (precondition for multi-lane).

**Phase 3 — org cutover mechanics:**
13. OpenRouter first-pass marker + gate job in the org reusable, bound to the reviewed head (no carry-forward across force-pushes); follow-up spend gating; timeout finalizer; lane-appropriate `max_diff_kb`.
14. Run the gate advisory (non-required) during the test window. At cutover, the ruleset swap — requiring the OpenRouter gate where desired and removing `review / grok-first-pass-gate` — is a separate, explicitly Nathan-approved manual operation, never an implementation changeset. Then retire the Grok thin callers and org reusable, and eventually the `XAI_API_KEY` secret. Advance the stale `v1` aliases after verified releases. Decide RetireBench's (currently absent) protection separately.

**Post-cutover — the destination:** multi-model lanes and bake-offs (unblocked by item 12), agent personas (the reserved hook), judge hardening.

## Changes unlikely to close the gap

- Raising Grok `max_turns` (Grok is frozen regardless).
- Treating Grok's `max_turns=50` and OpenRouter's `max_tool_turns=50` as equivalent units.
- Forcing one arbitrary tool call (the `tool_choice=required` nudge only fires on *zero* tool calls; it cannot prevent shallow two-tool glances).
- Repeating the workflow/README-specific blast-radius prompt for every kind of pull request (the coverage manifest subsumes it).
- Adding lanes or a judge to recover findings no lane discovered.
- Adding personas before the single-lane loop is reliable.
- Making OpenRouter a required check before parity is demonstrated.
- Drawing any harness-vs-harness conclusion from the v1.0.0 runs (#311, Pro #210).

## Test-window measurement

- Per-turn prompt size, latency, and tool-observation bytes — not only aggregate totals — plus findings quality on the same PRs, side by side with Grok.
- The posted review already sums per-lane `prompt_tokens + completion_tokens` and elapsed time; compare against Grok runs on the same PRs.
- Check the OpenRouter dashboard for xAI cached-input passthrough: it determines whether the resend snowball is a cost problem or only a latency one, which calibrates how much compaction work is worth.

## Verification performed

- [Codex] Grok harness tests: 148 passed, 2 skipped. OpenRouter harness tests: 89 passed. Upstream branch, tag, org reusable, and caller SHAs checked directly.
- [Claude] Full static read of all three repos at the cited SHAs; live verification of thin-caller pins (`RetireGolden/RetireGolden` → `.github@54bde42f` / `.github@fe20f331`), tag targets, and the branch-protection rulesets in the gate table above.
- No code, action, workflow, pull request, or external repository state was changed by either review.
