# Repository review guidance

`REVIEW.md` describes the contracts a reviewer should check. It is optional;
prose-only files work. Its guidance is additive to caller instructions and the
normal review sweep. It cannot authorize execution, choose arbitrary providers,
hide supported bugs, exempt files, or change merge and CI requirements.

## Enable and bootstrap

Use a pinned action revision that supports `review_policy: base`, a full-depth
checkout containing the PR head and target commit, Git supporting `--no-lazy-fetch`,
and the direct `role: all`
action. Enable the input in both initial and verification steps. Policy loading
defaults to off, and this release rejects policy-enabled setup/lane/judge roles.

Land the initial guidance using the existing review process. Its own PR uses
the previous target-branch policy. On the next normal PR, confirm the review
receipt names the expected target SHA and guidance files. A local preview does
not make model calls; a normal PR review uses the configured API account.

## Hierarchy and source

For `src/storage/reader.py`, the applicable files are root `REVIEW.md`,
`src/REVIEW.md`, and `src/storage/REVIEW.md`, when present. Sibling guidance is
scoped to its own changed files. Parent guidance remains applicable when a
child adds detail; conflicting prose should be reported as ambiguity.

Discovery walks only the repository root and ancestor directories of changed,
carried, and rename/deletion paths. For each such directory it lists immediate
children from the immutable target-branch tree (one bounded `git ls-tree` per
unique directory, deduplicated). Irrelevant subtrees are not traversed, and
unrelated unsupported sibling names are ignored. Each directory listing and the
Git `diff --name-status` inventory is capped at 4 MiB of output; an oversized
required listing or inventory is an error. The inventory still includes both
sides of renames, deleted paths, and carried findings from earlier rounds before
diff truncation. The verification
prompt can use an incremental diff without forgetting those obligations.
Reading adjacent source through tools does not change the frozen policy or
reviewer roster.

For a rename, the original directory's guidance applies to the old side and
transition, and the destination directory's guidance applies to the new side.
Both participate in the review. Moving a file does not permanently transfer
its old directory's contracts to its new location.

The loader reads Git blobs from the immutable target-branch tip captured during
collection. This is separate from the merge base and previous reviewed commit.
It never loads authoritative policy from the PR checkout's working files. New
or edited head policy is a proposal to review, not authority for that review.
All reviewers and the judge share the saved snapshot.

Use the exact case `REVIEW.md`. Applicable case variants, symlinks, submodule
traversal, unsafe paths, missing commit objects, invalid configuration, or limit
overflow produce an error before paid requests. Missing optional files use the
normal baseline. The loader never silently drops applicable guidance.

## Metadata

An optional fenced JSON block must begin the file (UTF-8 BOM is accepted).
The remaining content is Markdown. Use exactly one `review-policy` block:

````markdown
```review-policy
{
  "version": 1,
  "review": {"profile": "code", "minimum": "standard"}
}
```

# Review guidance

Keep old stored documents readable after an upgrade. Follow migrations and
restore paths as well as the changed writer. See docs/storage.md.
````

| Setting | Contract |
| --- | --- |
| `version` | Required integer `1` when metadata exists |
| `review` | Required object; may be empty |
| `review.profile` | Root-only profile slug; this release accepts `code` and `docs` as descriptive categories and keeps workflow model settings |
| `review.minimum` | `standard` or `deep`; omitted means standard |
| `review.rules` | Optional array of `{id, paths, minimum}`; IDs unique within a file |
| `paths` | Nonempty relative globs; `*` and `?` stay within one component, `**` crosses directories |

Unknown keys, duplicate JSON keys, wrong types, malformed fences, unsupported
versions, non-finite numbers, and nested profile overrides are errors. Rules
are relative to their containing directory. Child minima and matched rules can
only raise the inherited minimum; they cannot lower it.

The parser and offline preview understand `deep` so configurations can be
validated before rollout. **The current guidance-only action refuses a matching
deep request** because it has no configured deep execution profile. It never
silently runs the standard roster and calls that a deep review. Do not enable
such a rule in production until profile execution is available.

Example for offline planning only:

```json
{
  "version": 1,
  "review": {
    "rules": [
      {"id": "format-change", "paths": ["migrations/**", "crypto.ts"], "minimum": "deep"}
    ]
  }
}
```

No expressions, shell commands, templates, remote includes, credentials,
model/provider slugs, or spending limits belong in file metadata. Those remain
trusted workflow configuration.

## Bounds

| Resource | Limit |
| --- | --- |
| One file | 16 KiB UTF-8 |
| Applicable unique files combined | 64 KiB and 32 files |
| Ancestor traversal | 20 levels |
| Rules combined | 64 |
| Globs | 16 per rule, 256 combined, 256 UTF-8 bytes each |
| Git inventory output | 4 MiB per `ls-tree` directory listing and `diff --name-status` inventory |

Keep guidance concise and specific. Link local authoritative documentation;
the reviewer can read relevant evidence using its existing bounded tools.
Cross-repository and remote links are not automatically fetched. Do not assume
the review checkout contains sibling repositories or private organization docs.

## Local validation

Install this action's Python package in a separate trusted environment, then
run against the repository being reviewed:

```bash
python -P -m or_pr_review policy lint --repo . REVIEW.md src/storage/REVIEW.md
python -P -m or_pr_review policy explain --repo . --base FULL_TARGET_SHA --head FULL_PR_SHA
```

Supply full immutable 40-character SHAs. All required Git objects must already
be local. `-P` prevents the inspected checkout from shadowing the installed
Python package. `explain` reports source files/blob IDs, scopes, matched rules, and
effective policy digest, without printing prose. Add `--carried-path path/to/file`
to preview a carried finding. `lint` validates individual proposed files;
`explain` checks their effective ancestry and aggregate bounds. Neither command
uses GitHub credentials, network requests, or OpenRouter. Missing objects in a
partial clone fail locally; Git lazy fetching is disabled. For `lint`, file paths
must stay inside the chosen `--repo` root, and nested root-only profile settings
are rejected as well as syntax errors.

To preview a proposed policy as authoritative after merge, use a temporary local
commit containing that policy as the base and a subsequent example change as
the head. This does not alter what the current PR review trusts.

## Existing instructions, receipts, and recovery

Keep contribution and process instructions in `AGENTS.md`. Put review-specific
contracts in `REVIEW.md`, with references rather than duplicated manuals.
`custom_instructions` and `path_profiles` remain additive caller-owned settings.
Do not interpolate repository file contents into those trusted inputs.

The review body reports policy source SHA and digest after the existing header
fields. Lane artifacts retain the complete frozen policy. They contain repository
content and must remain trusted same-repository artifacts. Policy follows the
same configured OpenRouter/provider processing as other review context; it is
not a place for secrets or unrelated confidential material.

Publication context version 2 adds policy provenance. Use the same action
revision for artifact producers and consumers. Older version 1 artifacts require
a rerun; ledger version 1 remains supported, preserving finding IDs and replies.
Use `review_mode: auto` for a full-PR manual recheck that retains the ledger.

This first release adds guidance rather than a new authorization gate. A later
base-branch change does not mutate an active review; a new run resolves a new
snapshot. Extra-review requests, profile completion gates, and artifact reuse
are separate features and are not implied by enabling guidance.

| Symptom | Resolution |
| --- | --- |
| New guidance did not apply to its own PR | Expected: target-branch guidance is authoritative; verify the next PR after merge |
| Policy object missing | Ensure full-depth checkout contains both immutable commits |
| Unknown profile or matching deep rule | Use supported guidance-only settings; do not remove required checks just to get a green run |
| Configuration error | Correct the reported file/key or bounds; use local lint and explain before review |
| Matrix mode rejected | Use direct `role: all` for guidance in this release |
| Artifact version mismatch | Rerun lanes with one compatible action revision, keeping the existing findings ledger |
