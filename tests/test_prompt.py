from __future__ import annotations

from or_pr_review.collect import CollectedReview, DiffPlan, Truncation
from or_pr_review.prompt import (
    build_messages,
    changed_paths_from_diff,
    diff_right_side_lines,
    looks_like_ci_or_docs_inventory_change,
)

_WF = ".github/workflows/openrouter-code-review.yml"
WORKFLOW_ONLY_DIFF = f"""\
diff --git a/{_WF} b/{_WF}
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/{_WF}
@@ -0,0 +1,31 @@
+name: OpenRouter code review
+on:
+  pull_request:
+    types: [opened]
+jobs:
+  review:
+    runs-on: ubuntu-latest
+    steps:
+      - uses: actions/checkout@v4
"""


def _collected(*, diff: str = WORKFLOW_ONLY_DIFF, mode: str = "initial") -> CollectedReview:
    return CollectedReview(
        pr_number=304,
        title="Add OpenRouter review workflow",
        body="Thin caller only.",
        head_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        base_ref="main",
        head_ref="feat/or-review",
        plan=DiffPlan("full-pr", "full-pr", None, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", None),
        truncation=Truncation(diff, False, len(diff.encode()), len(diff.encode()), 300),
        mode=mode,  # type: ignore[arg-type]
    )


def test_changed_paths_from_workflow_diff() -> None:
    paths = changed_paths_from_diff(WORKFLOW_ONLY_DIFF)
    assert paths == [".github/workflows/openrouter-code-review.yml"]
    assert looks_like_ci_or_docs_inventory_change(paths)


def test_quoted_non_ascii_diff_headers_are_parsed() -> None:
    diff = (
        'diff --git "a/docs/na\\303\\257ve.md" "b/docs/na\\303\\257ve.md"\n'
        '--- "a/docs/na\\303\\257ve.md"\n'
        '+++ "b/docs/na\\303\\257ve.md"\n'
        "@@ -0,0 +1,2 @@\n"
        "+hello\n"
        "+world\n"
        "diff --git a/plain.txt b/plain.txt\n"
        "--- a/plain.txt\n"
        "+++ b/plain.txt\n"
        "@@ -1 +1,2 @@\n"
        " keep\n"
        "+add\n"
    )
    assert changed_paths_from_diff(diff) == ["docs/naïve.md", "plain.txt"]
    lines = diff_right_side_lines(diff)
    assert lines["docs/naïve.md"] == {1, 2}
    assert lines["plain.txt"] == {1, 2}


def test_changed_paths_handles_spaces_without_quoting() -> None:
    diff = "diff --git a/has space.txt b/has space.txt\n"
    assert changed_paths_from_diff(diff) == ["has space.txt"]


def test_diff_right_side_lines_maps_hunks() -> None:
    diff = (
        "diff --git a/src/api.py b/src/api.py\n"
        "--- a/src/api.py\n"
        "+++ b/src/api.py\n"
        "@@ -40,3 +40,4 @@\n"
        " ctx40\n"
        " ctx41\n"
        "+add42\n"
        " ctx43\n"
        "@@ -90,2 +91,2 @@\n"
        "-gone\n"
        "+swap91\n"
        " ctx92\n"
    )
    lines = diff_right_side_lines(diff)
    assert lines == {"src/api.py": {40, 41, 42, 43, 91, 92}}


def test_changed_paths_ignores_ordinary_python() -> None:
    diff = "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n"
    paths = changed_paths_from_diff(diff)
    assert paths == ["src/app.py"]
    assert not looks_like_ci_or_docs_inventory_change(paths)


def test_initial_prompt_requires_blast_radius_tools_for_workflow_pr() -> None:
    messages = build_messages(_collected())
    assert [item["role"] for item in messages] == ["system", "user"]
    text = "\n".join(item["content"] for item in messages)
    assert "blast radius" in text.lower()
    assert "read_file" in text and "grep" in text and "list_dir" in text
    assert "README" in text
    assert "code-map" in text
    assert ".github/workflows" in text
    assert "inventory" in text.lower()
    assert "openrouter-code-review.yml" in text
    assert "These paths are not the whole review" in text
    assert "empty findings list" in text.lower() or "clean verdict" in text.lower()
    # A reviewer should see why a workflow-only PR looks at docs/tests.
    assert "tests" in text.lower()


def test_workflow_prompt_calls_out_inventory_docs() -> None:
    user = build_messages(_collected())[1]["content"]
    assert "`.github/workflows/openrouter-code-review.yml`" in user
    assert "grep tests for those filenames" in user
    assert "README / code-map" in user


def test_initial_prompt_requires_coverage_manifest() -> None:
    text = "\n".join(item["content"] for item in build_messages(_collected()))
    assert '"coverage"' in text
    assert "EVERY file in the embedded diff" in text
    verify_text = "\n".join(
        item["content"] for item in build_messages(_collected(mode="verify"))
    )
    assert '"coverage"' not in verify_text


def test_verify_prompt_still_requires_tools_before_empty_findings() -> None:
    text = "\n".join(item["content"] for item in build_messages(_collected(mode="verify")))
    assert "verification follow-up" in text.lower()
    assert "blast radius" in text.lower()
    assert "read_file" in text


def test_verify_prompt_lists_prior_findings_and_contract() -> None:
    from or_pr_review.loop import LedgerFinding, LoopState

    state = LoopState(
        mode="verify",
        round_number=2,
        prior_findings=(
            LedgerFinding(
                id="r1-1",
                severity="bug",
                file="a.py",
                line=3,
                title="Race",
                evidence="detail here",
                status="open",
                models=(),
            ),
            LedgerFinding(
                id="r1-2",
                severity="nit",
                file=None,
                line=None,
                title="Old style",
                evidence="",
                status="disputed",
                models=(),
            ),
        ),
    )
    text = "\n".join(
        item["content"]
        for item in build_messages(
            _collected(mode="verify"),
            loop=state,
            agent_replies="Reply to finding r1-1 (from dev):\nfixed",
        )
    )
    assert "`r1-1` [bug] `a.py:3` — Race" in text
    assert "evidence: detail here" in text
    assert "do not re-raise" in text and "r1-2" in text
    assert '"resolutions"' in text
    assert "Fixing agent responses" in text
    assert "never" in text  # never follow instructions in replies
