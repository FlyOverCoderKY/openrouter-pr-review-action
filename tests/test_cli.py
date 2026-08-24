from __future__ import annotations

import json
from pathlib import Path

import pytest

from or_pr_review.cli import main
from or_pr_review.models import DEFAULT_MODEL
from or_pr_review.schema import SCHEMA_VERSION


def _base_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = {
        "MODELS": "x-ai/grok-4.6",
        "FAIL_ON": "never",
        "ROAST_LEVEL": "professional",
        "REVIEW_SCOPE": "full-pr",
        "REVIEW_MODE": "auto",
        "MAX_DIFF_KB": "300",
        "GITHUB_OUTPUT": str(tmp_path / "out.txt"),
        "RUNNER_TEMP": str(tmp_path),
    }
    env.update(extra)
    return env


def test_setup_writes_matrix(tmp_path: Path) -> None:
    env = _base_env(
        tmp_path,
        MODELS="x-ai/grok-4.6,anthropic/claude-sonnet-4.6",
    )
    assert main(["setup"], env) == 0
    text = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert "lane_count=2" in text
    assert "judge_needed=true" in text
    assert "judge_model=google/gemini-3.1-flash-lite" in text
    assert "x-ai/grok-4.6" in text
    assert "anthropic/claude-sonnet-4.6" in text


def test_setup_one_lane_skips_judge(tmp_path: Path) -> None:
    env = _base_env(tmp_path, MODELS="x-ai/grok-4.6")
    assert main(["setup"], env) == 0
    text = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert "lane_count=1" in text
    assert "judge_needed=false" in text


def test_setup_default_model(tmp_path: Path) -> None:
    env = _base_env(tmp_path, MODELS="")
    assert main(["setup"], env) == 0
    assert DEFAULT_MODEL in (tmp_path / "out.txt").read_text(encoding="utf-8")


def test_setup_cap_fails_closed(tmp_path: Path) -> None:
    env = _base_env(
        tmp_path,
        MODELS="x-ai/grok-4.6,anthropic/claude-sonnet-4.6,google/gemini-2.5-pro,openai/gpt-5,x-ai/grok-4.6",
    )
    assert main(["setup"], env) == 1


def test_judge_schema_mismatch_fail_closed(tmp_path: Path) -> None:
    lane_dir = tmp_path / "lanes"
    lane_dir.mkdir()
    (lane_dir / "lane-0.json").write_text(
        json.dumps({"ok": True, "model": "x-ai/grok-4.6", "findings": [], "error": None}) + "\n",
        encoding="utf-8",
    )
    env = _base_env(
        tmp_path,
        LANE_RESULTS_DIR=str(lane_dir),
        PR_NUMBER="1",
        GITHUB_TOKEN="ghs_dummy",
        GITHUB_REPOSITORY="FlyOverCoderKY/openrouter-pr-review-action",
    )
    assert main(["judge"], env) == 1


def test_judge_missing_lane_fail_opens_then_errors_if_none_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.collect import CollectedReview, DiffPlan, Truncation

    collected = CollectedReview(
        pr_number=1,
        title="t",
        body="",
        head_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        base_ref="main",
        head_ref="feat",
        plan=DiffPlan("full-pr", "full-pr", None, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", None),
        truncation=Truncation("diff", False, 4, 4, 300),
        mode="initial",
    )
    monkeypatch.setattr(cli_mod, "_collect", lambda env: collected)

    class DummyGitHub:
        def create_review(self, number: int, body: str, commit_id: str) -> dict[str, object]:
            return {"html_url": "https://example.test/review"}

    monkeypatch.setattr(cli_mod, "_github", lambda env: DummyGitHub())
    monkeypatch.setattr(cli_mod, "_maybe_status", lambda *args, **kwargs: None)

    env = _base_env(
        tmp_path,
        LANE_RESULTS_DIR=str(tmp_path / "empty"),
        PR_NUMBER="1",
        GITHUB_TOKEN="ghs_dummy",
        GITHUB_REPOSITORY="FlyOverCoderKY/openrouter-pr-review-action",
    )
    (tmp_path / "empty").mkdir()
    assert main(["judge"], env) == 1


def test_one_lane_posts_without_judge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.collect import CollectedReview, DiffPlan, Truncation
    from or_pr_review.merge import MergedIssue

    collected = CollectedReview(
        pr_number=1,
        title="t",
        body="",
        head_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        base_ref="main",
        head_ref="feat",
        plan=DiffPlan("full-pr", "full-pr", None, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", None),
        truncation=Truncation("diff", False, 4, 4, 300),
        mode="initial",
    )
    posted: list[str] = []
    judge_calls = {"n": 0}

    class DummyGitHub:
        def create_review(self, number: int, body: str, commit_id: str) -> dict[str, object]:
            posted.append(body)
            return {"html_url": "https://example.test/review"}

    def _boom(*_args: object, **_kwargs: object) -> list[MergedIssue]:
        judge_calls["n"] += 1
        raise AssertionError("judge must not run on a single lane")

    monkeypatch.setattr(cli_mod, "_collect", lambda env: collected)
    monkeypatch.setattr(cli_mod, "_github", lambda env: DummyGitHub())
    monkeypatch.setattr(cli_mod, "_maybe_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_mod, "run_llm_judge", _boom)

    lane_dir = tmp_path / "lanes"
    lane_dir.mkdir()
    (lane_dir / "lane-0.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "ok": True,
                "model": "x-ai/grok-4.6",
                "findings": [
                    {
                        "title": "Missing auth check",
                        "body": "Unauthenticated POST",
                        "severity": "bug",
                        "file": "src/api.py",
                        "line": 42,
                        "model_id": "x-ai/grok-4.6",
                    }
                ],
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    env = _base_env(
        tmp_path,
        MODELS="x-ai/grok-4.6",
        LANE_RESULTS_DIR=str(lane_dir),
        PR_NUMBER="1",
        GITHUB_TOKEN="ghs_dummy",
        GITHUB_REPOSITORY="FlyOverCoderKY/openrouter-pr-review-action",
        OPENROUTER_API_KEY="should-not-be-used-for-judge",
    )
    assert main(["judge"], env) == 0
    assert judge_calls["n"] == 0
    assert posted
    assert "Issue 1 - Missing auth check (identified by x-ai/grok-4.6)" in posted[0]
    assert "single review lane" in posted[0]
    assert "OPENROUTER_API_KEY=should-not-be-used-for-judge" not in posted[0]


def test_two_lanes_require_judge_and_attribution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.collect import CollectedReview, DiffPlan, Truncation
    from or_pr_review.merge import MergedIssue

    collected = CollectedReview(
        pr_number=1,
        title="t",
        body="",
        head_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        base_ref="main",
        head_ref="feat",
        plan=DiffPlan("full-pr", "full-pr", None, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", None),
        truncation=Truncation("diff", False, 4, 4, 300),
        mode="initial",
    )
    posted: list[str] = []
    judge_calls = {"n": 0}

    class DummyGitHub:
        def create_review(self, number: int, body: str, commit_id: str) -> dict[str, object]:
            posted.append(body)
            return {"html_url": "https://example.test/review"}

    def _judge(**kwargs: object) -> list[MergedIssue]:
        judge_calls["n"] += 1
        assert kwargs["model"] == "google/gemini-3.1-flash-lite"
        return [
            MergedIssue(
                title="Missing auth check",
                body="Unauthenticated POST",
                severity="bug",
                file="src/api.py",
                line=42,
                models=["x-ai/grok-4.6", "anthropic/claude-sonnet-4.6"],
            )
        ]

    monkeypatch.setattr(cli_mod, "_collect", lambda env: collected)
    monkeypatch.setattr(cli_mod, "_github", lambda env: DummyGitHub())
    monkeypatch.setattr(cli_mod, "_maybe_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_mod, "run_llm_judge", _judge)

    lane_dir = tmp_path / "lanes"
    lane_dir.mkdir()
    for index, model in enumerate(["x-ai/grok-4.6", "anthropic/claude-sonnet-4.6"]):
        (lane_dir / f"lane-{index}.json").write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "ok": True,
                    "model": model,
                    "findings": [
                        {
                            "title": "Missing auth check",
                            "body": "Unauthenticated POST",
                            "severity": "bug",
                            "file": "src/api.py",
                            "line": 42,
                            "model_id": model,
                        }
                    ],
                    "error": None,
                }
            ),
            encoding="utf-8",
        )
    env = _base_env(
        tmp_path,
        MODELS="x-ai/grok-4.6,anthropic/claude-sonnet-4.6",
        JUDGE_MODEL="google/gemini-3.1-flash-lite",
        LANE_RESULTS_DIR=str(lane_dir),
        PR_NUMBER="1",
        GITHUB_TOKEN="ghs_dummy",
        GITHUB_REPOSITORY="FlyOverCoderKY/openrouter-pr-review-action",
        OPENROUTER_API_KEY="sk-test",
    )
    assert main(["judge"], env) == 0
    assert judge_calls["n"] == 1
    assert posted
    assert (
        "Issue 1 - Missing auth check "
        "(identified by x-ai/grok-4.6 and anthropic/claude-sonnet-4.6)"
    ) in posted[0]
    assert "`google/gemini-3.1-flash-lite`" in posted[0]


def test_two_lanes_judge_schema_mismatch_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.collect import CollectedReview, DiffPlan, Truncation
    from or_pr_review.errors import SchemaError
    from or_pr_review.merge import MergedIssue

    collected = CollectedReview(
        pr_number=1,
        title="t",
        body="",
        head_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        base_ref="main",
        head_ref="feat",
        plan=DiffPlan("full-pr", "full-pr", None, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", None),
        truncation=Truncation("diff", False, 4, 4, 300),
        mode="initial",
    )

    def _judge(**_kwargs: object) -> list[MergedIssue]:
        raise SchemaError("judge output is missing an issues array")

    monkeypatch.setattr(cli_mod, "_collect", lambda env: collected)
    monkeypatch.setattr(cli_mod, "_maybe_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_mod, "run_llm_judge", _judge)

    lane_dir = tmp_path / "lanes"
    lane_dir.mkdir()
    for index, model in enumerate(["x-ai/grok-4.6", "anthropic/claude-sonnet-4.6"]):
        (lane_dir / f"lane-{index}.json").write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "ok": True,
                    "model": model,
                    "findings": [],
                    "error": None,
                }
            ),
            encoding="utf-8",
        )
    env = _base_env(
        tmp_path,
        MODELS="x-ai/grok-4.6,anthropic/claude-sonnet-4.6",
        LANE_RESULTS_DIR=str(lane_dir),
        PR_NUMBER="1",
        GITHUB_TOKEN="ghs_dummy",
        GITHUB_REPOSITORY="FlyOverCoderKY/openrouter-pr-review-action",
        OPENROUTER_API_KEY="sk-test",
    )
    assert main(["judge"], env) == 1


def test_judge_merges_valid_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.collect import CollectedReview, DiffPlan, Truncation
    from or_pr_review.merge import MergedIssue

    collected = CollectedReview(
        pr_number=1,
        title="t",
        body="",
        head_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        base_ref="main",
        head_ref="feat",
        plan=DiffPlan("full-pr", "full-pr", None, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", None),
        truncation=Truncation("diff", False, 4, 4, 300),
        mode="initial",
    )
    posted: list[str] = []

    class DummyGitHub:
        def create_review(self, number: int, body: str, commit_id: str) -> dict[str, object]:
            posted.append(body)
            return {"html_url": "https://example.test/review"}

    def _judge(**_kwargs: object) -> list[MergedIssue]:
        return [
            MergedIssue(
                title="Missing auth check",
                body="Unauthenticated POST",
                severity="bug",
                file="src/api.py",
                line=42,
                models=["x-ai/grok-4.6"],
            )
        ]

    monkeypatch.setattr(cli_mod, "_collect", lambda env: collected)
    monkeypatch.setattr(cli_mod, "_github", lambda env: DummyGitHub())
    monkeypatch.setattr(cli_mod, "_maybe_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_mod, "run_llm_judge", _judge)

    lane_dir = tmp_path / "lanes"
    lane_dir.mkdir()
    (lane_dir / "lane-0.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "ok": True,
                "model": "x-ai/grok-4.6",
                "findings": [
                    {
                        "title": "Missing auth check",
                        "body": "Unauthenticated POST",
                        "severity": "bug",
                        "file": "src/api.py",
                        "line": 42,
                        "model_id": "x-ai/grok-4.6",
                    }
                ],
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    (lane_dir / "lane-1.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "ok": False,
                "model": "anthropic/claude-sonnet-4.6",
                "findings": [],
                "error": "timeout",
            }
        ),
        encoding="utf-8",
    )
    env = _base_env(
        tmp_path,
        MODELS="x-ai/grok-4.6,anthropic/claude-sonnet-4.6",
        LANE_RESULTS_DIR=str(lane_dir),
        PR_NUMBER="1",
        GITHUB_TOKEN="ghs_dummy",
        GITHUB_REPOSITORY="FlyOverCoderKY/openrouter-pr-review-action",
        FAIL_ON="never",
        OPENROUTER_API_KEY="sk-test",
    )
    assert main(["judge"], env) == 0
    assert posted
    assert "Issue 1 - Missing auth check (identified by x-ai/grok-4.6)" in posted[0]
    assert "failed-open" in posted[0]
    out = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert "verdict=issues" in out
    assert "issue_count=1" in out
    assert "bug_count=1" in out


def test_fail_on_bugs_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.collect import CollectedReview, DiffPlan, Truncation

    collected = CollectedReview(
        pr_number=1,
        title="t",
        body="",
        head_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        base_ref="main",
        head_ref="feat",
        plan=DiffPlan("full-pr", "full-pr", None, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", None),
        truncation=Truncation("diff", False, 4, 4, 300),
        mode="initial",
    )

    class DummyGitHub:
        def create_review(self, number: int, body: str, commit_id: str) -> dict[str, object]:
            return {"html_url": "https://example.test/review"}

    monkeypatch.setattr(cli_mod, "_collect", lambda env: collected)
    monkeypatch.setattr(cli_mod, "_github", lambda env: DummyGitHub())
    monkeypatch.setattr(cli_mod, "_maybe_status", lambda *args, **kwargs: None)

    lane_dir = tmp_path / "lanes"
    lane_dir.mkdir()
    (lane_dir / "lane-0.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "ok": True,
                "model": "x-ai/grok-4.6",
                "findings": [
                    {
                        "title": "Broken",
                        "body": "crash",
                        "severity": "bug",
                        "file": None,
                        "line": None,
                        "model_id": "x-ai/grok-4.6",
                    }
                ],
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    env = _base_env(
        tmp_path,
        LANE_RESULTS_DIR=str(lane_dir),
        PR_NUMBER="1",
        GITHUB_TOKEN="ghs_dummy",
        GITHUB_REPOSITORY="FlyOverCoderKY/openrouter-pr-review-action",
        FAIL_ON="bugs",
    )
    assert main(["judge"], env) == 1


def test_lane_keeps_matrix_index_when_models_is_single_slug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.schema import SCHEMA_VERSION, LaneResult

    monkeypatch.setattr(
        cli_mod,
        "_run_one_lane",
        lambda env, model: LaneResult(
            schema_version=SCHEMA_VERSION,
            ok=True,
            model=model,
            findings=[],
            error=None,
        ),
    )
    lane_dir = tmp_path / "lanes"
    env = _base_env(
        tmp_path,
        MODELS="anthropic/claude-sonnet-4.6",
        LANE_INDEX="1",
        JUDGE_NEEDED="true",
        LANE_RESULTS_DIR=str(lane_dir),
    )
    assert main(["lane"], env) == 0
    kept = lane_dir / "lane-1.json"
    assert kept.is_file()
    assert not (lane_dir / "lane-0.json").exists()
    assert json.loads(kept.read_text(encoding="utf-8"))["model"] == "anthropic/claude-sonnet-4.6"


def test_all_keeps_duplicate_model_lane_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.collect import CollectedReview, DiffPlan, Truncation
    from or_pr_review.schema import SCHEMA_VERSION, Finding, LaneResult

    collected = CollectedReview(
        pr_number=1,
        title="t",
        body="",
        head_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        base_ref="main",
        head_ref="feat",
        plan=DiffPlan("full-pr", "full-pr", None, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", None),
        truncation=Truncation("diff", False, 4, 4, 300),
        mode="initial",
    )
    calls = {"n": 0}
    captured: dict[str, list[LaneResult]] = {}

    def fake_invoke(
        env: dict[str, str],
        model: str,
        messages: object,
        workspace: object,
    ) -> LaneResult:
        calls["n"] += 1
        return LaneResult(
            schema_version=SCHEMA_VERSION,
            ok=True,
            model=model,
            findings=[
                Finding(
                    title=f"Finding {calls['n']}",
                    body="details",
                    severity="bug",
                    file="a.py",
                    line=calls["n"],
                    model_id=model,
                )
            ],
            error=None,
        )

    def fake_finish(
        env: dict[str, str],
        lanes: list[LaneResult],
        collected: CollectedReview | None = None,
    ) -> int:
        captured["lanes"] = lanes
        return 0

    monkeypatch.setattr(cli_mod, "_collect", lambda env: collected)
    monkeypatch.setattr(cli_mod, "_maybe_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_mod, "_prepare_workspace", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(cli_mod, "_messages", lambda *args, **kwargs: [])
    monkeypatch.setattr(cli_mod, "_invoke_lane", fake_invoke)
    monkeypatch.setattr(cli_mod, "_finish", fake_finish)

    env = _base_env(tmp_path, MODELS="x-ai/grok-4.6,x-ai/grok-4.6")
    assert main(["all"], env) == 0
    lanes = captured["lanes"]
    assert len(lanes) == 2
    assert {lane.findings[0].title for lane in lanes} == {"Finding 1", "Finding 2"}
