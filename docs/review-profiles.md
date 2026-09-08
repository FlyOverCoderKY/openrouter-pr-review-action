# Trusted review profiles

`review_profiles` is trusted workflow configuration. It is a version-1 JSON
registry of named `standard` and optional `deep` panels; it is not repository
policy and it does not grant merge authority.

## Contract

```yaml
with:
  review_profiles: >-
    {"version":1,"profiles":{"code":{"standard":{"lanes":[
    {"model":"x-ai/grok-4.6","required":true},
    {"model":"z-ai/glm-5.3-flash","required":false}],
    "effort":"high","verify_effort":"low","max_tool_turns":50,
    "verify_max_tool_turns":30,"lane_timeout_seconds":600,
    "job_budget_seconds":1320},"deep":{"lanes":[
    {"model":"x-ai/grok-4.6","required":true},
    {"model":"openai/gpt-6-astra","required":true},
    {"model":"z-ai/glm-5.3-flash","required":false}],
    "effort":"high","verify_effort":"low","max_tool_turns":50,
    "verify_max_tool_turns":30,"lane_timeout_seconds":600,
    "job_budget_seconds":1320}}}}
  review_level: auto
```

The example keeps the standing Grok + optional GLM standard illustration and
adds Astra only as a deep illustration. It is not an organization default or a
fresh model recommendation. Use exact model slugs available in your account;
`model_routes` is a separate mapping keyed by those exact slugs.

The registry is at most 32 KiB, eight profiles, and four lanes per panel. Every
panel needs at least one `required: true` lane. A deep panel retains every
required standard lane and cannot lower explicit effort, tool, or time
resources. `effort` may be omitted (empty) to delegate to the provider
default; standard and deep must either both omit or both explicitly set each
effort field (`effort` and `verify_effort`) when comparing levels. 
`verify_effort` and `verify_max_tool_turns` apply to verification rounds.
Each panel may set `judge_model`, independent of its lane slugs. `model_routes`
is a separate exact-slug mapping for lanes and does not configure the judge.
`max_tool_turns`, `job_budget_seconds`, and `lane_timeout_seconds` are profile
budgets subject to the caller's ceilings (the action defaults the lane ceiling
to 1080 seconds). The panel's lane timeout must leave the 180-second
publication reserve.

When a registry is supplied, do not combine it with the caller inputs `models`, `judge_model`,
`judge_needed`, or `effort`. The legacy path remains available: without a
registry, standard lanes are synthesized from the existing inputs and retain
the CLI defaults (Grok reviewer and Luna judge); a requested deep review fails
before paid work unless a deep mapping exists.

The root `REVIEW.md` profile may be descriptive `code` or `docs`, or a custom
profile name present in the trusted registry. Child rules can raise the
minimum but cannot select models or routes.

`review_level: auto` follows the effective target-branch policy minimum;
`deep` raises the request. Deep retains the finding ledger and continues a
full-PR review. `initial` is the explicit reset; `auto` seeds an initial/full
PR review when no ledger exists. A partial panel cannot advance the ledger.

## Prepared matrix execution

All-role execution collects one frozen snapshot (PR head, policy, replies,
model plan, and absolute deadline) and shares it across the lanes and judge.
It preserves the context and receipt as artifacts for 30 days. For a matrix,
run setup first and download its context artifact in each lane and judge job.
Lanes and judge must receive the
expected setup digest and the resolved head SHA; they do not independently
collect policy. Even a one-lane prepared matrix publishes through a judge job.
There is no additive lane cache: the entire panel reruns.

The digest detects mismatches; it is not authenticity. Download only the setup
artifact from the same workflow run and pin the identical action revision in
setup, lane, and judge jobs. Queue time consumes the total deadline.

Use `role: all` when possible. If using a matrix, pass setup outputs
(`head_sha`, `registry_digest`, `review_context_sha256`, and the artifact name)
through frozen context and set the lane matrix strategy to `fail-fast: false`.
Run the judge with `if: always()` after setup succeeds; the judge
must consume the frozen context rather than model strings copied from the
matrix.

When `review_context_file` and `review_context_sha256` are supplied together,
`role: all` loads that frozen snapshot and does not recollect policy or loop
state. `role: setup` may write the snapshot to `review_context_output_file`
when an external orchestrator needs a fixed destination path.

## Outputs

Prepared runs expose `review_profile`, `review_level`, `review_trigger`,
`registry_digest`, `profile_satisfied`, `panel_status`, `review_context_sha256`,
`review_context_artifact`, `review_receipt_artifact`,
and `head_sha`. Required-lane absence yields
`required_missing`/partial or error and a non-zero result even with
`fail_on: never`; optional-lane failure is visible as degraded. Policy
publication context is version 3; the finding ledger remains version 1.

## Authorization boundary

Action outputs alone do not stop an old standard review from authorizing a
pending deep request. A trusted request handler must mark the PR pending before
accepting it, require the profile status, validate workflow provenance and the
current effective policy, retain pending status if a label is removed, and
support explicit cancellation and base-policy refresh. Repository integration
for that handler is not implemented here: a `review:deep` label by itself does
not enforce pending requests, and dispatching `review_level: deep` is only an
action input.

`REVIEW.md` should never name models, provider secrets, or merge authority.
Full-depth review is inert source checkout only; the action does not execute PR
code.
