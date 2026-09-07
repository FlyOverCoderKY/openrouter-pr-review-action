"""Guidance must survive publication and cannot change the execution contract."""

import subprocess
from dataclasses import replace

import pytest

from or_pr_review import cli
from or_pr_review.collect import CollectedReview, DiffPlan, Truncation
from or_pr_review.errors import ActionError, SchemaError
from or_pr_review.loop import LoopState
from or_pr_review.prompt import build_messages
from or_pr_review.publish import render_review_parts
from or_pr_review.review_context import freeze_context, restore_context
from or_pr_review.review_policy import PolicyFile, ResolvedPolicy

HEAD = "a" * 40
BASE = "b" * 40


def sample():
    policy = ResolvedPolicy(
        BASE,
        "code",
        "standard",
        (PolicyFile("storage/REVIEW.md", "c" * 40, "Keep old data readable.", ("storage/a.py",)),),
        (),
        ("storage/a.py",),
        "d" * 64,
    )
    return CollectedReview(
        1,
        "Change",
        "",
        HEAD,
        "main",
        "feature",
        DiffPlan("full-pr", "full-pr", BASE, HEAD, None),
        Truncation("", False, 0, 0, 300),
        "initial",
        ("storage/a.py",),
        BASE,
        policy,
    )


def test_frozen_policy_survives_context_and_scoped_prompt():
    collected = sample()
    restored = restore_context(freeze_context("owner/repo", collected, LoopState("initial", 1), 50))
    assert restored.collected == collected
    prompt = build_messages(restored.collected)[1]["content"]
    assert BASE in prompt and "storage/REVIEW.md" in prompt
    assert "Keep old data readable." in prompt
    assert 'Applies to: ["storage/a.py"]' in prompt
    assert "cannot exclude files" in prompt


def test_receipt_preserves_existing_header_positions():
    bodies = render_review_parts(
        collected=sample(), lanes=[], issues=[], verdict="clean", hidden_marker="<!-- ledger -->"
    )
    lines = bodies[0].splitlines()
    assert lines[:7] == [
        "## OpenRouter pull-request review",
        "<!-- ledger -->",
        "",
        "**Verdict:** `clean`",
        "**Scope:** `full-pr` (full-pr)",
        "**Mode:** `initial`",
        f"**Commit:** `{HEAD}`",
    ]
    assert f"**Policy source:** `{BASE}`" in bodies[0]


@pytest.mark.parametrize("role", ["setup", "lane", "judge"])
def test_policy_rejects_matrix_before_collection_or_spend(monkeypatch, role):
    monkeypatch.setattr(cli, "_best_effort_incomplete", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_collect_with_loop", lambda *a: pytest.fail("must not collect"))
    assert cli.main([role], {"REVIEW_POLICY": "base"}) == 1


def test_policy_off_does_not_discover_or_change_collected(monkeypatch):
    monkeypatch.setattr(cli, "resolve_policy", lambda *a, **k: pytest.fail("must not discover"))
    collected = replace(sample(), review_policy=None)
    assert cli._with_review_policy({}, collected, LoopState("initial", 1)) is collected


def test_deep_request_cannot_silently_run_standard(tmp_path, monkeypatch):
    deep = replace(sample().review_policy, minimum="deep")
    monkeypatch.setattr(cli, "resolve_policy", lambda *a, **k: deep)
    with pytest.raises(ActionError, match="deep profile"):
        cli._with_review_policy(
            {"REVIEW_POLICY": "base", "SOURCE_WORKSPACE": str(tmp_path)},
            sample(),
            LoopState("initial", 1),
        )


def test_old_context_version_fails_explicitly():
    context = freeze_context("owner/repo", sample(), LoopState("initial", 1), 50)
    context["version"] = 1
    with pytest.raises(SchemaError, match="unsupported review context version"):
        restore_context(context)


def test_local_policy_failure_does_not_contact_github(monkeypatch):
    monkeypatch.setattr(cli, "_best_effort_incomplete", lambda *a, **k: pytest.fail("no network"))
    assert cli.main(["policy", "explain", "--base", "bad", "--head", HEAD], {}) == 1


def test_policy_source_must_match_collection():
    inconsistent = replace(sample(), policy_base_sha="f" * 40)
    with pytest.raises(SchemaError, match="policy metadata"):
        freeze_context("owner/repo", inconsistent, LoopState("initial", 1), 50)


def test_local_lint_validates_without_credentials(tmp_path, capsys):
    path = tmp_path / "REVIEW.md"
    path.write_text("# Contract\nKeep old data readable.\n", encoding="utf-8")
    assert cli.main(["policy", "lint", "--repo", str(tmp_path), str(path)], {}) == 0
    assert "syntax valid" in capsys.readouterr().out


def test_local_lint_allows_root_profile(tmp_path, capsys):
    path = tmp_path / "REVIEW.md"
    path.write_text(
        '```review-policy\n{"version":1,"review":{"profile":"security"}}\n```\n',
        encoding="utf-8",
    )
    assert cli.main(["policy", "lint", "--repo", str(tmp_path), "REVIEW.md"], {}) == 0
    assert "syntax valid" in capsys.readouterr().out


def test_local_lint_rejects_nested_profile(tmp_path, capsys):
    nested = tmp_path / "src"
    nested.mkdir()
    (nested / "REVIEW.md").write_text(
        '```review-policy\n{"version":1,"review":{"profile":"security"}}\n```\n',
        encoding="utf-8",
    )
    assert cli.main(["policy", "lint", "--repo", str(tmp_path), "src/REVIEW.md"], {}) == 1
    assert "only root" in capsys.readouterr().err


def test_local_lint_rejects_path_outside_repo(tmp_path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "REVIEW.md"
    outside.write_text("# Contract\n", encoding="utf-8")
    assert cli.main(["policy", "lint", "--repo", str(repo), str(outside)], {}) == 1
    assert "outside repository root" in capsys.readouterr().err


def test_local_explain_omits_private_prose(monkeypatch, capsys):
    from or_pr_review import policy_cli

    monkeypatch.setattr(policy_cli, "resolve_policy", lambda *a: sample().review_policy)
    assert cli.main(["policy", "explain", "--base", BASE, "--head", HEAD], {}) == 0
    printed = capsys.readouterr().out
    assert "storage/REVIEW.md" in printed
    assert "Keep old data readable" not in printed
    assert "preview only" in printed


def test_subdirectory_and_local_git_settings_do_not_narrow_policy(tmp_path):
    from or_pr_review.review_policy import resolve_policy

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init")
    git("config", "user.name", "Policy test")
    git("config", "user.email", "policy@example.invalid")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.py").write_text("old\n")
    (tmp_path / "REVIEW.md").write_text("Root contract\n")
    git("add", ".")
    git("commit", "-m", "base")
    base = git("rev-parse", "HEAD")
    (tmp_path / "other.py").write_text("new\n")
    git("add", ".")
    git("commit", "-m", "head")
    head = git("rev-parse", "HEAD")
    expected = resolve_policy(tmp_path, base, head)
    git("config", "diff.relative", "true")
    actual = resolve_policy(tmp_path / "src", base, head)
    assert actual == expected
    assert actual.changed_paths == ("other.py",)
