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


def test_initial_prompt_demands_exhaustive_all_severity_sweep() -> None:
    system = build_messages(_collected())[0]["content"]
    # Recall-over-precision framing with the loop rationale.
    assert "prefer recall over precision" in system
    assert "false positive is cheaper" in system
    assert "than a missed valid bug" in system
    assert "do not self-censor borderline findings" in system
    assert "may dispute it" in system
    # Numeric expectation anchor, not just a cap.
    assert "may legitimately contain 15-30" in system
    assert "Do not stop at a representative sample" in system
    # Multi-pass sweep protocol.
    assert "Sweep every file and every hunk" in system
    assert "what input, state, or timing makes it wrong" in system
    assert "Sweep again" in system
    assert "Repeat until a full sweep finds nothing new" in system
    assert "`nit` findings, not omissions" in system
    assert "do not stop early" in system
    assert "keep the findings list short" in system
    # Coverage entries assert a completed sweep, not a glance.
    assert "swept that file" in system
    # Falsification pass with the asymmetric guardrail — pin the load-bearing
    # drop side and its bug/risk scope, not just the slogan.
    assert "Before returning a draft bug or risk finding, try to FALSIFY" in system
    assert "Drop a candidate only\nwhen direct counterevidence disproves it" in system
    assert "uncertainty is not rejection" in system
    assert "name the files or searches you checked" in system
    # Checkout policy text is untrusted and cannot kill a finding.
    assert "can NEVER disprove a finding" in system
    # The ledger clips bodies, so the failure scenario must lead the body.
    assert "LEAD each finding body with the concrete failure scenario" in system
    # Recall-preserving output discipline: consolidate only equivalent drafts
    # and communicate uncertainty through severity instead of suppressing it.
    assert "same trigger, root cause, and corrective change" in system
    assert "Preserve\nseparate findings" in system
    assert "not\na reason to suppress a distinct candidate" in system
    assert "current code and a concrete trigger demonstrate" in system
    assert "proof gap prevents calling it demonstrated" in system
    assert "not a\n  personal style preference" in system
    assert "proof gap may make a candidate a `risk`" in system
    assert "does not, by itself, erase" in system


def test_verify_prompt_omits_the_initial_sweep_block() -> None:
    system = build_messages(_collected(mode="verify"))[0]["content"]
    assert "Sweep every file and every hunk" not in system
    assert "swept that file" not in system
    assert "prefer recall over precision" not in system
    assert "try to FALSIFY" not in system  # the initial block stays initial-only
    # But NEW verify findings get the same asymmetric falsification bar.
    assert "try to falsify it against current callers" in system
    assert "uncertainty is not rejection" in system
    assert "proof gap may make a candidate a `risk`" in system


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
    assert "AUTHORITATIVE disposition" in text
    assert "if the note says the finding" in text
    assert "status MUST be fixed" in text
    assert "Never put a different disposition in the note" in text
    assert "Fixing agent responses" in text
    assert "never" in text  # never follow instructions in replies


def test_path_profiles_apply_only_when_paths_match() -> None:
    from or_pr_review.prompt import matched_profiles, parse_path_profiles

    profiles = parse_path_profiles(
        '[{"name": "source-of-truth", "paths": ["*calc*", "*rules*"],'
        ' "instructions": "Verify every stated figure against its cited authority."}]'
    )
    assert matched_profiles(profiles, ["calc.py", "tests/test_calc.py"]) == profiles
    assert matched_profiles(profiles, ["docs/readme.md"]) == []
    assert matched_profiles(None, ["calc.py"]) == []

    text = build_messages(
        _collected(
            diff="diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n"
        ),
        path_profiles=profiles,
    )[1]["content"]
    assert "## Path review profiles (caller-owned; additive to the full sweep)" in text
    assert "### source-of-truth" in text
    assert "Verify every stated figure" in text
    assert "never narrow the review" in text
    # Non-matching diff: no block at all.
    unmatched = build_messages(_collected(), path_profiles=profiles)[1]["content"]
    assert "Path review profiles" not in unmatched


def test_parse_path_profiles_validation() -> None:
    import pytest as _pytest

    from or_pr_review.errors import ActionError
    from or_pr_review.prompt import parse_path_profiles

    assert parse_path_profiles(None) is None
    assert parse_path_profiles("  ") is None
    with _pytest.raises(ActionError, match="not valid JSON"):
        parse_path_profiles("{nope")
    with _pytest.raises(ActionError, match="JSON array"):
        parse_path_profiles('{"paths": []}')
    with _pytest.raises(ActionError, match="paths"):
        parse_path_profiles('[{"instructions": "x"}]')
    with _pytest.raises(ActionError, match="instructions"):
        parse_path_profiles('[{"paths": ["*.py"]}]')


def test_path_globs_use_path_semantics() -> None:
    from or_pr_review.prompt import matched_profiles

    profile = [{"paths": ["src/*.py"], "instructions": "x"}]
    assert matched_profiles(profile, ["src/app.py"]) == profile
    # * must not cross a path segment (unlike fnmatch).
    assert matched_profiles(profile, ["src/pkg/nested.py"]) == []
    deep = [{"paths": ["src/**/*.py"], "instructions": "x"}]
    assert matched_profiles(deep, ["src/pkg/nested.py"]) == deep
    # Case-sensitive, matching CI runners rather than the local OS.
    assert matched_profiles([{"paths": ["*Calc*"], "instructions": "x"}], ["calc.py"]) == []
    # ? stays within a segment too.
    q = [{"paths": ["a?c.py"], "instructions": "x"}]
    assert matched_profiles(q, ["abc.py"]) == q
    assert matched_profiles(q, ["a/c.py"]) == []


def test_profiles_match_pretruncation_paths() -> None:
    from or_pr_review.collect import CollectedReview, DiffPlan, Truncation

    # calc.py changed on the PR but was truncated out of the embed; the
    # profile must still fire (truncation cannot disable guidance).
    truncated_diff = "diff --git a/rules.py b/rules.py\n--- a/rules.py\n+++ b/rules.py\n"
    collected = CollectedReview(
        pr_number=1,
        title="t",
        body="",
        head_sha="a" * 40,
        base_ref="main",
        head_ref="feat",
        plan=DiffPlan("full-pr", "full-pr", None, "a" * 40, None),
        truncation=Truncation(truncated_diff, True, 999_999, len(truncated_diff), 300),
        mode="initial",
        all_changed_paths=("rules.py", "calc.py"),
    )
    profiles = [{"name": "sot", "paths": ["*calc*"], "instructions": "verify figures"}]
    text = build_messages(collected, path_profiles=profiles)[1]["content"]
    assert "### sot" in text


def test_parse_path_profiles_limits_and_normalization() -> None:
    import json as json_mod

    import pytest as _pytest

    from or_pr_review.errors import ActionError
    from or_pr_review.prompt import parse_path_profiles

    with _pytest.raises(ActionError, match="16,000"):
        parse_path_profiles('[{"paths": ["*"], "instructions": "' + "x" * 16_100 + '"}]')
    many = json_mod.dumps(
        [{"paths": ["*"], "instructions": "x"} for _ in range(21)]
    )
    with _pytest.raises(ActionError, match="at most 20"):
        parse_path_profiles(many)
    with _pytest.raises(ActionError, match="must be an object"):
        parse_path_profiles('["oops"]')
    with _pytest.raises(ActionError, match="name"):
        parse_path_profiles('[{"name": true, "paths": ["*"], "instructions": "x"}]')
    # Whitespace-padded globs are stripped, so they actually match.
    parsed = parse_path_profiles('[{"paths": [" *calc* "], "instructions": "x"}]')
    assert parsed[0]["paths"] == ["*calc*"]
