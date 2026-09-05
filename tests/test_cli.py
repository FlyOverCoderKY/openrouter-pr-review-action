from __future__ import annotations

import json
from concurrent.futures import TimeoutError as FutureTimeoutError
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
    assert "judge_model=openai/gpt-5.6-luna" in text
    assert "x-ai/grok-4.6" in text
    assert "anthropic/claude-sonnet-4.6" in text


def test_setup_one_lane_skips_judge(tmp_path: Path) -> None:
    env = _base_env(tmp_path, MODELS="x-ai/grok-4.6")
    assert main(["setup"], env) == 0
    text = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert "lane_count=1" in text
    assert "judge_needed=false" in text


@pytest.mark.parametrize("role", ["setup", "all"])
def test_full_roster_rejects_unmatched_routes(tmp_path, monkeypatch, role):
    from or_pr_review import cli

    monkeypatch.setattr(cli, "_collect_with_loop", lambda *_: pytest.fail("must validate first"))
    env = _base_env(tmp_path, MODEL_ROUTES='{"openai/gpt-6-astr":{"service_tier":"flex"}}')
    assert main([role], env) == 1


def test_single_matrix_lane_accepts_sibling_routes(tmp_path):
    from or_pr_review import cli

    env = _base_env(tmp_path, MODEL_ROUTES='{"openai/gpt-6-astra":{"service_tier":"flex"}}')
    assert cli._validate_inputs(env) == ["x-ai/grok-4.6"]


@pytest.mark.parametrize(
    "raw",
    [
        " " * 8_000 + "{}",
        json.dumps({f"openai/model-{i}": {"service_tier": "flex"} for i in range(5)}),
        '{" openai/gpt-6-astra":{"service_tier":"flex"}}',
    ],
)
def test_model_route_limits(raw):
    from or_pr_review.errors import ActionError
    from or_pr_review.models import parse_model_routes

    with pytest.raises(ActionError):
        parse_model_routes(raw)


@pytest.mark.parametrize("model", ["openai/gpt-6-astra", "x-ai/grok-4.6", "z-ai/glm-5.3-flash"])
def test_model_route_reaches_only_its_lane(tmp_path, monkeypatch, model):
    from or_pr_review import cli
    from or_pr_review.schema import failed_lane

    captured = {}

    def run(**kwargs):
        captured.update(kwargs)
        return failed_lane(model, "test")

    monkeypatch.setattr(cli, "run_lane", run)
    env = _base_env(
        tmp_path,
        OPENROUTER_API_KEY="test-key",
        MODEL_ROUTES=json.dumps(
            {"openai/gpt-6-astra": {"provider": "openai/flex", "service_tier": "flex"}}
        ),
    )
    cli._invoke_lane(env, model, [], tmp_path)
    if model == "openai/gpt-6-astra":
        assert captured["provider_order"] == ["openai/flex"]
        assert captured["service_tier"] == "flex"
    else:
        assert "provider_order" not in captured
        assert "service_tier" not in captured


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        "null",
        "broken",
        '{"bad-slug": {"service_tier": "flex"}}',
        '{"openai/gpt-6-astra": {"service_tier": []}}',
        '{"openai/gpt-6-astra": {"service_tier": "typo"}}',
        '{"openai/gpt-6-astra": {"provider": ""}}',
        '{"openai/gpt-6-astra": {"allow_fallbacks": true}}',
    ],
)
def test_invalid_model_route_fails_before_collection(tmp_path, monkeypatch, raw):
    from or_pr_review import cli

    monkeypatch.setattr(cli, "_collect_with_loop", lambda *_: pytest.fail("must validate first"))
    assert main(["all"], _base_env(tmp_path, MODEL_ROUTES=raw)) == 1


def test_setup_honors_explicit_judge_override_once(tmp_path: Path) -> None:
    env = _base_env(
        tmp_path,
        MODELS="x-ai/grok-4.6,anthropic/claude-sonnet-4.6",
        JUDGE_NEEDED="false",
    )
    assert main(["setup"], env) == 0
    text = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert text.count("judge_needed=") == 1
    assert "judge_needed=false" in text


@pytest.mark.parametrize(
    ("name", "value"),
    [("LANE_MODEL", "not-a-slug"), ("OPENROUTER_TIMEOUT_SECONDS", "0")],
)
def test_lane_rejects_invalid_model_and_timeout(tmp_path: Path, name: str, value: str) -> None:
    env = _base_env(tmp_path, **{name: value})
    assert main(["lane"], env) == 1


def test_invalid_status_comments_does_not_escape_error_handler(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = _base_env(tmp_path, STATUS_COMMENTS="maybe", PR_NUMBER="1")

    assert main(["lane"], env) == 1
    captured = capsys.readouterr()
    assert "status_comments must be true or false" in captured.err
    assert "Traceback" not in captured.err


def test_invalid_all_role_deadline_is_rejected_before_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import cli as cli_mod

    monkeypatch.setattr(
        cli_mod,
        "_collect_with_loop",
        lambda _env: pytest.fail("collection must not run before input validation"),
    )
    env = _base_env(tmp_path, ALL_ROLE_DEADLINE_SECONDS="0")

    assert main(["all"], env) == 1


def test_setup_default_model(tmp_path: Path) -> None:
    env = _base_env(tmp_path, MODELS="")
    assert main(["setup"], env) == 0
    assert DEFAULT_MODEL in (tmp_path / "out.txt").read_text(encoding="utf-8")


def test_setup_default_max_tool_turns_is_fifty(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    env.pop("MAX_TOOL_TURNS", None)
    assert main(["setup"], env) == 0


@pytest.mark.parametrize("value", ["abc", "0"])
def test_invalid_job_budget_is_reported_as_an_action_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    env = _base_env(tmp_path, JOB_BUDGET_SECONDS=value)
    assert main(["lane"], env) == 1
    captured = capsys.readouterr()
    assert "job_budget_seconds" in captured.err.lower()
    assert "Traceback" not in captured.err


def test_anchor_checkout_requires_the_reviewed_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An available commit object is insufficient when HEAD is elsewhere."""
    from or_pr_review import cli as cli_mod

    calls: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = "b" * 40 + "\n"

    def fake_run(cmd: list[str], **_kwargs: object) -> Result:
        calls.append(cmd)
        return Result()

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    assert not cli_mod._checkout_has_commit(tmp_path, "a" * 40)
    assert calls[0][-3:] == ["rev-parse", "--verify", "HEAD"]

    Result.stdout = "A" * 40 + "\n"
    assert cli_mod._checkout_has_commit(tmp_path, "a" * 40)


@pytest.mark.parametrize("value", ["line one\nline two", "line one\rline two"])
def test_set_output_rejects_multiline_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.errors import ActionError

    output = tmp_path / "output.txt"
    monkeypatch.setattr(cli_mod, "_ACTIVE_ENV", {"GITHUB_OUTPUT": str(output)})

    with pytest.raises(ActionError, match="must be a single line"):
        cli_mod._set_output("unsafe", value)
    assert not output.exists()


def test_capped_union_note_is_visible_to_review_readers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from or_pr_review import cli as cli_mod

    monkeypatch.setattr(cli_mod, "deterministic_union_with_cap", lambda _lanes: ([], 3))
    issues, note = cli_mod._capped_union_note([], "deadline fallback")
    assert issues == []
    assert note == "deadline fallback (capped+3)"


def test_setup_rejects_out_of_range_max_tool_turns(tmp_path: Path) -> None:
    env = _base_env(tmp_path, MAX_TOOL_TURNS="-1")
    assert main(["setup"], env) == 1
    env = _base_env(tmp_path, MAX_TOOL_TURNS="1001")
    assert main(["setup"], env) == 1


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


def test_judge_missing_lane_fail_opens_then_errors_if_none_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

        def pr_view(self, number: int) -> dict[str, object]:
            return {"headRefOid": "a" * 40}

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

        def pr_view(self, number: int) -> dict[str, object]:
            return {"headRefOid": "a" * 40}

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
    assert "Issue 1 — Missing auth check" in posted[0]
    assert "identified by x-ai/grok-4.6" in posted[0]
    assert "single review lane" in posted[0]
    assert "OPENROUTER_API_KEY=should-not-be-used-for-judge" not in posted[0]
    outputs = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert outputs.count("judge_needed=") == 1
    assert "judge_needed=false" in outputs
    assert outputs.count("judge_model=") == 1


def test_diagnostic_only_lane_posts_visible_partial_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.loop import LoopState
    from or_pr_review.schema import Finding, LaneResult

    collected = _mk_collected()
    posted: list[str] = []

    class DummyGitHub:
        def create_review(self, number: int, body: str, commit_id: str) -> dict[str, object]:
            posted.append(body)
            return {"html_url": "https://example.test/review"}

        def pr_view(self, number: int) -> dict[str, object]:
            return {"headRefOid": collected.head_sha}

    lane = LaneResult(
        schema_version=SCHEMA_VERSION,
        ok=True,
        model="x-ai/grok-4.6",
        findings=[
            Finding(
                title="File unavailable in the provided review checkout",
                body="The review tooling could not read or inspect the supplied snapshot.",
                severity="risk",
                file="src/file.py",
                line=None,
                model_id="x-ai/grok-4.6",
            )
        ],
        error=None,
        head_sha=collected.head_sha,
    )
    monkeypatch.setattr(cli_mod, "_github", lambda env: DummyGitHub())
    monkeypatch.setattr(cli_mod, "_maybe_status", lambda *args, **kwargs: None)

    env = _base_env(
        tmp_path,
        PR_NUMBER="1",
        GITHUB_TOKEN="ghs_dummy",
        GITHUB_REPOSITORY="FlyOverCoderKY/openrouter-pr-review-action",
    )
    monkeypatch.setattr(cli_mod, "_ACTIVE_ENV", env)
    cli_mod._write_lane_setup_outputs(["x-ai/grok-4.6"])
    assert (
        cli_mod._finish(
            env,
            [lane],
            collected=collected,
            loop=LoopState(mode="initial", round_number=1),
        )
        == 0
    )
    assert "**Verdict:** `partial`" in posted[0]
    assert "review environment" in posted[0]
    assert "must not be treated as a clean pass" in posted[0]
    outputs = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert outputs.count("judge_needed=") == 1
    assert outputs.count("judge_model=") == 1


def test_two_lanes_require_judge_and_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.collect import CollectedReview, DiffPlan, Truncation
    from or_pr_review.merge import MergedIssue

    reviewed_sha = "a" * 40
    source = tmp_path / "reviewed"
    (source / "src").mkdir(parents=True)
    (source / "src/api.py").write_text("unsafe_call()\n", encoding="utf-8")
    diff = (
        "diff --git a/src/api.py b/src/api.py\n"
        "--- /dev/null\n"
        "+++ b/src/api.py\n"
        "@@ -0,0 +1 @@\n"
        "+unsafe_call()\n"
    )
    collected = CollectedReview(
        pr_number=1,
        title="t",
        body="",
        head_sha=reviewed_sha,
        base_ref="main",
        head_ref="feat",
        plan=DiffPlan("full-pr", "full-pr", None, reviewed_sha, None),
        truncation=Truncation(diff, False, len(diff), len(diff), 300),
        mode="initial",
    )
    posted: list[str] = []
    inline_comments: list[dict[str, object]] = []
    judge_calls = {"n": 0}

    class DummyGitHub:
        def create_review(
            self,
            number: int,
            body: str,
            commit_id: str,
            comments: list[dict[str, object]] | None = None,
        ) -> dict[str, object]:
            posted.append(body)
            inline_comments.extend(comments or [])
            return {"html_url": "https://example.test/review"}

        def pr_view(self, number: int) -> dict[str, object]:
            return {"headRefOid": reviewed_sha}

    def _judge(**kwargs: object) -> list[MergedIssue]:
        judge_calls["n"] += 1
        assert kwargs["model"] == "google/gemini-3.1-flash-lite"
        return (
            [
                MergedIssue(
                    title="Missing auth check",
                    body="Unauthenticated POST",
                    severity="bug",
                    file="src/api.py",
                    line=1,
                    models=["x-ai/grok-4.6", "anthropic/claude-sonnet-4.6"],
                )
            ],
            "merged",
            0.0021,
        )

    monkeypatch.setattr(cli_mod, "_collect", lambda env: collected)
    monkeypatch.setattr(cli_mod, "_github", lambda env: DummyGitHub())
    monkeypatch.setattr(cli_mod, "_maybe_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_mod, "run_llm_judge", _judge)
    monkeypatch.setattr(
        cli_mod.subprocess,
        "run",
        lambda *_args, **_kwargs: type(
            "Result", (), {"returncode": 0, "stdout": reviewed_sha + "\n"}
        )(),
    )

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
                            "line": 1,
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
        SOURCE_WORKSPACE=str(source),
    )
    assert main(["judge"], env) == 0
    assert judge_calls["n"] == 1
    assert posted
    assert "Issue 1 — Missing auth check" in posted[0]
    assert "identified by x-ai/grok-4.6 and anthropic/claude-sonnet-4.6" in posted[0]
    assert "`google/gemini-3.1-flash-lite`" in posted[0]
    # The production wiring must render the judge cost on the posted body
    # (the lane artifacts in this test carry no cost, so the sum is
    # labeled incomplete rather than posing as the run total).
    assert "**Cost:** $0.0021" in posted[0]
    assert "incomplete: no cost reported for" in posted[0]
    assert inline_comments
    assert inline_comments[0]["path"] == "src/api.py"
    assert inline_comments[0]["line"] == 1


def test_two_lanes_judge_schema_mismatch_uses_validated_union(
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
    posted: list[str] = []

    class DummyGitHub:
        def create_review(self, number: int, body: str, commit_id: str) -> dict[str, object]:
            posted.append(body)
            return {"html_url": "https://example.test/review"}

        def pr_view(self, number: int) -> dict[str, object]:
            return {"headRefOid": "a" * 40}

    def _judge(**_kwargs: object) -> list[MergedIssue]:
        raise SchemaError("judge output is missing an issues array")

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
        LANE_RESULTS_DIR=str(lane_dir),
        PR_NUMBER="1",
        GITHUB_TOKEN="ghs_dummy",
        GITHUB_REPOSITORY="FlyOverCoderKY/openrouter-pr-review-action",
        OPENROUTER_API_KEY="sk-test",
    )
    assert main(["judge"], env) == 0
    assert posted
    assert "schema fallback: deterministic union" in posted[0]
    assert "Missing auth check" in posted[0]


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

        def pr_view(self, number: int) -> dict[str, object]:
            return {"headRefOid": "a" * 40}

    def _judge(**_kwargs: object) -> list[MergedIssue]:
        raise AssertionError("one successful lane must bypass the judge")

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
    assert "Issue 1 — Missing auth check" in posted[0]
    assert "identified by x-ai/grok-4.6" in posted[0]
    assert "failed-open" in posted[0]
    out = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert "verdict=issues" in out
    assert "issue_count=1" in out
    assert "bug_count=1" in out
    assert out.count("judge_needed=") == 1
    assert "judge_needed=true" in out
    assert out.count("judge_model=") == 1


def test_explicit_judge_override_runs_with_one_successful_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.merge import MergedIssue
    from or_pr_review.schema import Finding, LaneResult

    lane = LaneResult(
        schema_version=SCHEMA_VERSION,
        ok=True,
        model="fast/model",
        findings=[
            Finding(
                title="Race",
                body="check then act",
                severity="bug",
                file="a.py",
                line=1,
                model_id="fast/model",
            )
        ],
        error=None,
    )
    calls = {"count": 0}

    def fake_judge(**_kwargs: object):
        calls["count"] += 1
        return (
            [
                MergedIssue(
                    title="Race",
                    body="check then act",
                    severity="bug",
                    file="a.py",
                    line=1,
                    models=["fast/model"],
                )
            ],
            "merged",
            0.001,
        )

    monkeypatch.setattr(cli_mod, "run_llm_judge", fake_judge)
    env = {
        "MODELS": "fast/model",
        "JUDGE_NEEDED": "true",
        "OPENROUTER_API_KEY": "sk-test",
    }

    outcome = cli_mod._resolve_issues(env, ["fast/model"], [lane], [lane])

    assert calls["count"] == 1
    assert outcome.ran is True
    assert outcome.issues[0].title == "Race"


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

        def pr_view(self, number: int) -> dict[str, object]:
            return {"headRefOid": "a" * 40}

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
        lambda env, model: (
            LaneResult(
                schema_version=SCHEMA_VERSION,
                ok=True,
                model=model,
                findings=[],
                error=None,
            ),
            None,
            None,
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


def test_invoke_lane_defaults_max_tool_turns_to_fifty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.harness import DEFAULT_MAX_TOOL_TURNS
    from or_pr_review.schema import SCHEMA_VERSION, LaneResult

    captured: dict[str, object] = {}

    def fake_run_lane(**kwargs: object) -> LaneResult:
        captured.update(kwargs)
        return LaneResult(
            schema_version=SCHEMA_VERSION,
            ok=True,
            model=str(kwargs.get("model") or ""),
            findings=[],
            error=None,
        )

    monkeypatch.setattr(cli_mod, "run_lane", fake_run_lane)
    env = _base_env(tmp_path, OPENROUTER_API_KEY="sk-test")
    env.pop("MAX_TOOL_TURNS", None)
    result = cli_mod._invoke_lane(env, "x-ai/grok-4.6", [], tmp_path)
    assert result.ok
    assert captured["max_tool_turns"] == DEFAULT_MAX_TOOL_TURNS == 50


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
        **_kwargs: object,
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
        **_kwargs: object,
    ) -> int:
        captured["lanes"] = lanes
        return 0

    monkeypatch.setattr(cli_mod, "_collect", lambda env: collected)
    monkeypatch.setattr(cli_mod, "_maybe_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_mod, "_prepare_workspace", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(cli_mod, "_messages", lambda *args, **kwargs: [])
    monkeypatch.setattr(cli_mod, "_invoke_lane", fake_invoke)
    monkeypatch.setattr(cli_mod, "_finish", fake_finish)

    env = _base_env(
        tmp_path,
        MODELS="x-ai/grok-4.6,x-ai/grok-4.6",
        PR_NUMBER="1",
        GITHUB_TOKEN="ghs_dummy",
        GITHUB_REPOSITORY="FlyOverCoderKY/openrouter-pr-review-action",
    )
    assert main(["all"], env) == 0
    lanes = captured["lanes"]
    assert len(lanes) == 2
    assert {lane.findings[0].title for lane in lanes} == {"Finding 1", "Finding 2"}
    assert all(lane.head_sha == "a" * 40 for lane in lanes)


def _mk_collected(head: str = "a" * 40, fallback_notice: str | None = None):
    from or_pr_review.collect import CollectedReview, DiffPlan, Truncation

    return CollectedReview(
        pr_number=1,
        title="t",
        body="",
        head_sha=head,
        base_ref="main",
        head_ref="feat",
        plan=DiffPlan("full-pr", "full-pr", None, head, fallback_notice),
        truncation=Truncation("diff", False, 4, 4, 300),
        mode="initial",
    )


def test_all_persists_completed_lane_before_bounded_sibling_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR358 regression: one straggler must not erase a completed sibling."""
    from or_pr_review import cli as cli_mod
    from or_pr_review.loop import LoopState
    from or_pr_review.schema import LaneResult, failed_lane

    collected = _mk_collected()
    captured: dict[str, object] = {}
    lane_timeouts: list[int] = []
    work = tmp_path / "work"
    artifact_dir = tmp_path / "surviving-lanes"

    def fake_invoke(
        env: dict[str, str],
        model: str,
        messages: object,
        workspace: object,
        **kwargs: object,
    ) -> LaneResult:
        lane_timeouts.append(int(kwargs["lane_timeout"]))
        if model == "slow/model":
            return failed_lane(model, "lane wall-clock deadline exhausted")
        return LaneResult(
            schema_version=SCHEMA_VERSION,
            ok=True,
            model=model,
            findings=[],
            error=None,
        )

    def fake_as_completed(futures: object, timeout: int | None = None):
        ordered = list(futures)
        yield ordered[0]
        # _role_all must persist the completed result before asking for the
        # next future, because GitHub can cancel at this exact point.
        assert (artifact_dir / "lane-0.json").is_file()
        captured["early_artifact"] = True
        captured["timed_out"] = True
        raise FutureTimeoutError()

    def fake_finish(
        env: dict[str, str],
        lanes: list[LaneResult],
        **_kwargs: object,
    ) -> int:
        captured["lanes"] = lanes
        return 0

    monkeypatch.setattr(
        cli_mod,
        "_collect_with_loop",
        lambda env: (collected, LoopState(mode="initial", round_number=1), ""),
    )
    monkeypatch.setattr(cli_mod, "_maybe_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_mod, "_prepare_workspace", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(cli_mod, "_messages", lambda *args, **kwargs: [])
    monkeypatch.setattr(cli_mod, "_invoke_lane", fake_invoke)
    monkeypatch.setattr(cli_mod, "as_completed", fake_as_completed)
    monkeypatch.setattr(cli_mod, "_finish", fake_finish)

    env = _base_env(
        tmp_path,
        MODELS="fast/model,slow/model",
        WORK=str(work),
        ALL_LANE_RESULTS_DIR=str(artifact_dir),
        PR_NUMBER="1",
        GITHUB_TOKEN="ghs_dummy",
        GITHUB_REPOSITORY="FlyOverCoderKY/openrouter-pr-review-action",
        ALL_ROLE_DEADLINE_SECONDS="1",
    )
    assert main(["all"], env) == 0
    assert captured["early_artifact"] is True
    assert captured["timed_out"] is True
    lanes = captured["lanes"]
    assert isinstance(lanes, list)
    assert lanes[0].ok
    assert not lanes[1].ok
    assert "wall-clock deadline" in (lanes[1].error or "")
    assert (artifact_dir / "lane-1.json").is_file()
    expected_reserve = (
        cli_mod.POST_RESERVE_SECONDS
        + (cli_mod.MAX_RATE_LIMIT_ATTEMPTS - 1) * cli_mod.MAX_RETRY_AFTER_SECONDS
        + cli_mod.JUDGE_SCHEDULING_MARGIN_SECONDS
        + cli_mod.MAX_RATE_LIMIT_ATTEMPTS * cli_mod.MIN_JUDGE_ATTEMPT_SECONDS
    )
    assert all(
        timeout <= cli_mod.JOB_BUDGET_SECONDS - expected_reserve for timeout in lane_timeouts
    )


def test_matrix_lane_deadline_includes_collection_and_workspace_prep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR358 regression for reusable-workflow role=lane jobs."""
    from or_pr_review import cli as cli_mod
    from or_pr_review.loop import LoopState
    from or_pr_review.schema import LaneResult

    collected = _mk_collected()
    now = {"value": 0.0}
    captured: dict[str, object] = {}
    lane_dir = tmp_path / "matrix-lanes"

    monkeypatch.setattr(cli_mod, "JOB_BUDGET_SECONDS", 20)
    monkeypatch.setattr(cli_mod.time, "monotonic", lambda: now["value"])

    def fake_collect(env: dict[str, str]):
        now["value"] = 8.0
        return collected, LoopState(mode="initial", round_number=1), ""

    def fake_workspace(*_args: object, **_kwargs: object) -> Path:
        now["value"] = 12.0
        return tmp_path

    def fake_invoke(
        env: dict[str, str],
        model: str,
        messages: object,
        workspace: object,
        **kwargs: object,
    ) -> LaneResult:
        captured.update(kwargs)
        return LaneResult(
            schema_version=SCHEMA_VERSION,
            ok=True,
            model=model,
            findings=[],
            error=None,
        )

    monkeypatch.setattr(cli_mod, "_collect_with_loop", fake_collect)
    monkeypatch.setattr(cli_mod, "_prepare_workspace", fake_workspace)
    monkeypatch.setattr(cli_mod, "_messages", lambda *args, **kwargs: [])
    monkeypatch.setattr(cli_mod, "_maybe_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_mod, "_invoke_lane", fake_invoke)

    env = _base_env(
        tmp_path,
        MODELS="fast/model,slow/model",
        LANE_INDEX="1",
        LANE_MODEL="slow/model",
        LANE_RESULTS_DIR=str(lane_dir),
        JUDGE_NEEDED="true",
        PR_NUMBER="1",
        GITHUB_TOKEN="ghs_dummy",
    )
    assert main(["lane"], env) == 0
    assert captured["lane_timeout"] == 8
    assert (lane_dir / "lane-1.json").is_file()


def test_judge_deadline_falls_back_instead_of_starting_long_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.schema import Finding, LaneResult

    lanes = [
        LaneResult(
            schema_version=SCHEMA_VERSION,
            ok=True,
            model=model,
            findings=[
                Finding(
                    title="Race",
                    body="check then act",
                    severity="bug",
                    file="a.py",
                    line=1,
                    model_id=model,
                )
            ],
            error=None,
        )
        for model in ("fast/model", "slow/model")
    ]
    monkeypatch.setattr(
        cli_mod,
        "run_llm_judge",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("judge must not start near the job deadline")
        ),
    )
    env = {
        "MODELS": "fast/model,slow/model",
        cli_mod._JOB_DEADLINE_KEY: str(
            cli_mod.time.monotonic() + cli_mod.POST_RESERVE_SECONDS + 60
        ),
    }
    issues, note, cost, ran = cli_mod._resolve_issues(
        env, ["fast/model", "slow/model"], lanes, lanes
    )
    assert issues
    assert "deadline fallback" in note
    assert cost is None
    assert ran is False


def test_judge_timeout_is_clipped_to_fit_all_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from or_pr_review import cli as cli_mod

    monkeypatch.setattr(cli_mod.time, "monotonic", lambda: 100.0)
    # 180s post + 180s worst Retry-After + 5s scheduling + 7*30s requests.
    env = {
        "OPENROUTER_TIMEOUT_SECONDS": "180",
        cli_mod._JOB_DEADLINE_KEY: str(100 + 180 + 180 + 5 + 210),
    }
    assert cli_mod._judge_request_timeout(env) == 30


def test_prepare_workspace_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.errors import ActionError

    def boom(source: object, sha: object, dest: object, **kwargs: object) -> None:
        raise ActionError("git archive failed")

    monkeypatch.setattr(cli_mod, "materialize_commit", boom)
    env = {"MAX_TOOL_TURNS": "50", "SOURCE_WORKSPACE": str(tmp_path)}
    with pytest.raises(ActionError, match="refusing a tool-less review"):
        cli_mod._prepare_workspace(env, _mk_collected(), tmp_path / "work")


def test_prepare_workspace_skips_when_tools_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import cli as cli_mod

    called = {"n": 0}

    def track(*_args: object) -> None:
        called["n"] += 1

    monkeypatch.setattr(cli_mod, "materialize_commit", track)
    env = {"MAX_TOOL_TURNS": "0", "SOURCE_WORKSPACE": str(tmp_path)}
    assert cli_mod._prepare_workspace(env, _mk_collected(), tmp_path / "work") is None
    assert called["n"] == 0


def test_stale_head_marks_review_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from or_pr_review import cli as cli_mod

    posted: list[tuple[str, str]] = []

    class DummyGitHub:
        def create_review(self, number: int, body: str, commit_id: str) -> dict[str, object]:
            posted.append((body, commit_id))
            return {"html_url": "https://example.test/review"}

        def pr_view(self, number: int) -> dict[str, object]:
            return {"headRefOid": "b" * 40}

    monkeypatch.setattr(cli_mod, "_collect", lambda env: _mk_collected())
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
                "findings": [],
                "error": None,
                "head_sha": "a" * 40,
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
    )
    assert main(["judge"], env) == 0
    body, commit_id = posted[0]
    assert commit_id == "a" * 40
    assert "pinned to commit" in body
    out = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert "verdict=partial" in out


def test_long_findings_lists_post_continuation_comments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import cli as cli_mod

    reviews: list[str] = []
    comments: list[str] = []

    class DummyGitHub:
        def create_review(self, number: int, body: str, commit_id: str) -> dict[str, object]:
            reviews.append(body)
            return {"html_url": "https://example.test/review"}

        def create_issue_comment(self, number: int, body: str) -> dict[str, object]:
            comments.append(body)
            return {"html_url": "https://example.test/comment"}

        def pr_view(self, number: int) -> dict[str, object]:
            return {"headRefOid": "a" * 40}

    monkeypatch.setattr(cli_mod, "_collect", lambda env: _mk_collected())
    monkeypatch.setattr(cli_mod, "_github", lambda env: DummyGitHub())
    monkeypatch.setattr(cli_mod, "_maybe_status", lambda *args, **kwargs: None)

    lane_dir = tmp_path / "lanes"
    lane_dir.mkdir()
    findings = [
        {
            "title": f"Finding number {n}",
            "body": "y" * 6000,
            "severity": "bug",
            "file": "src/api.py",
            "line": n,
            "model_id": "x-ai/grok-4.6",
        }
        for n in range(1, 21)
    ]
    (lane_dir / "lane-0.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "ok": True,
                "model": "x-ai/grok-4.6",
                "findings": findings,
                "error": None,
                "head_sha": "a" * 40,
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
    )
    assert main(["judge"], env) == 0
    assert len(reviews) == 1
    assert comments, "long findings lists must continue in comments, not truncate"
    joined = "\n".join(reviews + comments)
    for n in range(1, 21):
        assert f"— Finding number {n}" in joined


def test_initial_coverage_count_mismatch_posts_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.collect import CollectedReview, DiffPlan, Truncation

    diff = "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n"
    collected = CollectedReview(
        pr_number=1,
        title="t",
        body="",
        head_sha="a" * 40,
        base_ref="main",
        head_ref="feat",
        plan=DiffPlan("full-pr", "full-pr", None, "a" * 40, None),
        truncation=Truncation(diff, False, len(diff), len(diff), 300),
        mode="initial",
    )
    posted: list[str] = []

    class DummyGitHub:
        def create_review(self, number: int, body: str, commit_id: str) -> dict[str, object]:
            posted.append(body)
            return {"html_url": "https://example.test/review"}

        def pr_view(self, number: int) -> dict[str, object]:
            return {"headRefOid": "a" * 40}

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
                "findings": [],
                "error": None,
                "head_sha": "a" * 40,
                "coverage": [{"path": "src/app.py", "findings": 2}],
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
    )
    assert main(["judge"], env) == 0
    assert "claims 2 finding(s)" in posted[0]
    out = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert "verdict=clean" in out


def _prior_ledger_marker(repo: str) -> str:
    from or_pr_review.loop import Ledger, LedgerFinding, encode_ledger

    prior = Ledger(
        round_number=1,
        findings=(
            LedgerFinding(
                id="r1-1",
                severity="bug",
                file="src/api.py",
                line=42,
                title="Missing auth check",
                evidence="Unauthenticated POST is accepted",
                status="open",
                models=("x-ai/grok-4.6",),
            ),
        ),
        reviewed_sha="b" * 40,
        generation="1234567890ab",
    )
    return encode_ledger(prior, repo=repo, pr_number=1)


class _LoopGitHub:
    def __init__(self, marker: str) -> None:
        self.marker = marker
        self.posted: list[str] = []
        self.comments: list[list[dict]] = []

    def list_bot_review_bodies(self, number: int, bot_login: str) -> list[str]:
        assert bot_login == "github-actions[bot]"
        return [f"## OpenRouter pull-request review\n{self.marker}\n"]

    def list_finding_replies(
        self, number: int, *, generation: str = ""
    ) -> list[tuple[str, str, str]]:
        return [("r1-1", "dev", "added the check in abc123")]

    def list_recent_issue_comments(self, number: int, limit: int = 30) -> list[tuple[str, str]]:
        return []

    def pr_view(self, number: int) -> dict[str, object]:
        return {"headRefOid": "a" * 40}

    def create_review(
        self,
        number: int,
        body: str,
        commit_id: str,
        comments: list[dict] | None = None,
    ) -> dict[str, object]:
        self.posted.append(body)
        self.comments.append(comments or [])
        return {"html_url": "https://example.test/review"}


def _verify_env(tmp_path: Path, lane_dir: Path, repo: str) -> dict[str, str]:
    return _base_env(
        tmp_path,
        LANE_RESULTS_DIR=str(lane_dir),
        PR_NUMBER="1",
        GITHUB_TOKEN="ghs_dummy",
        GITHUB_REPOSITORY=repo,
        REVIEW_MODE="verify",
    )


def test_verify_round_folds_ledger_and_updates_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.loop import extract_ledger

    repo = "FlyOverCoderKY/openrouter-pr-review-action"
    github = _LoopGitHub(_prior_ledger_marker(repo))
    monkeypatch.setattr(cli_mod, "_collect", lambda env: _mk_collected())
    monkeypatch.setattr(cli_mod, "_github", lambda env: github)
    monkeypatch.setattr(cli_mod, "_maybe_status", lambda *args, **kwargs: None)

    lane_dir = tmp_path / "lanes"
    lane_dir.mkdir()
    (lane_dir / "lane-0.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "ok": True,
                "model": "x-ai/grok-4.6",
                "findings": [],
                "error": None,
                "head_sha": "a" * 40,
                "resolutions": [{"id": "r1-1", "status": "fixed", "note": "check added"}],
            }
        ),
        encoding="utf-8",
    )
    assert main(["judge"], _verify_env(tmp_path, lane_dir, repo)) == 0
    body = github.posted[0]
    assert "### Round 2 resolution" in body
    assert "✅" in body and "r1-1" in body
    updated = extract_ledger(body, repo=repo, pr_number=1)
    assert updated is not None
    assert updated.round_number == 2
    assert updated.reviewed_sha == "a" * 40
    assert updated.generation == "1234567890ab"  # generation carries across rounds
    assert updated.findings == ()
    out = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert "verdict=clean" in out
    assert "round=2" in out
    assert "issue_count=0" in out


def test_verify_round_carries_unfixed_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.loop import extract_ledger

    repo = "FlyOverCoderKY/openrouter-pr-review-action"
    github = _LoopGitHub(_prior_ledger_marker(repo))
    monkeypatch.setattr(cli_mod, "_collect", lambda env: _mk_collected())
    monkeypatch.setattr(cli_mod, "_github", lambda env: github)
    monkeypatch.setattr(cli_mod, "_maybe_status", lambda *args, **kwargs: None)

    lane_dir = tmp_path / "lanes"
    lane_dir.mkdir()
    (lane_dir / "lane-0.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "ok": True,
                "model": "x-ai/grok-4.6",
                "findings": [],
                "error": None,
                "head_sha": "a" * 40,
                "resolutions": [{"id": "r1-1", "status": "not_fixed", "note": "still reachable"}],
            }
        ),
        encoding="utf-8",
    )
    assert main(["judge"], _verify_env(tmp_path, lane_dir, repo)) == 0
    body = github.posted[0]
    assert "❌" in body
    updated = extract_ledger(body, repo=repo, pr_number=1)
    assert updated is not None
    assert [finding.id for finding in updated.findings] == ["r1-1"]
    assert updated.findings[0].status == "open"
    out = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert "verdict=issues" in out
    assert "issue_count=1" in out
    assert "bug_count=1" in out


def test_verify_prompt_excludes_replies_to_retired_nits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.loop import Ledger, LedgerFinding, encode_ledger

    repo = "FlyOverCoderKY/openrouter-pr-review-action"
    prior = Ledger(
        round_number=1,
        findings=(
            LedgerFinding(
                id="r1-1",
                severity="bug",
                file="src/api.py",
                line=42,
                title="Missing auth check",
                evidence="Unauthenticated POST is accepted",
                status="open",
                models=("x-ai/grok-4.6",),
            ),
            LedgerFinding(
                id="r1-2",
                severity="nit",
                file="src/api.py",
                line=7,
                title="Duplicated citation",
                evidence="the same reference twice",
                status="open",
                models=("x-ai/grok-4.6",),
            ),
        ),
        reviewed_sha="b" * 40,
        generation="1234567890ab",
    )

    class _RepliesGitHub(_LoopGitHub):
        def list_finding_replies(
            self, number: int, *, generation: str = ""
        ) -> list[tuple[str, str, str]]:
            return [
                ("r1-1", "dev", "auth check added"),
                ("r1-2", "dev", "citation deduplicated"),
            ]

    github = _RepliesGitHub(encode_ledger(prior, repo=repo, pr_number=1))
    monkeypatch.setattr(cli_mod, "_collect", lambda env: _mk_collected())
    monkeypatch.setattr(cli_mod, "_github", lambda env: github)
    env = _base_env(
        tmp_path,
        PR_NUMBER="1",
        GITHUB_TOKEN="ghs_dummy",
        GITHUB_REPOSITORY=repo,
        REVIEW_MODE="verify",
    )
    _collected, state, agent_replies = cli_mod._collect_with_loop(env)
    assert [finding.id for finding in state.prior_findings] == ["r1-1"]
    assert [finding.id for finding in state.retired_prior] == ["r1-2"]
    assert "r1-1" in agent_replies
    assert "auth check added" in agent_replies
    assert "r1-2" not in agent_replies
    assert "citation deduplicated" not in agent_replies


def test_verify_round_retires_carried_nits_via_severity_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.loop import Ledger, LedgerFinding, encode_ledger, extract_ledger

    repo = "FlyOverCoderKY/openrouter-pr-review-action"

    def _nit(ident: str) -> LedgerFinding:
        return LedgerFinding(
            id=ident,
            severity="nit",
            file="src/api.py",
            line=7,
            title="Duplicated citation",
            evidence="the same reference twice",
            status="open",
            models=("x-ai/grok-4.6",),
        )

    prior = Ledger(
        round_number=1,
        findings=(
            LedgerFinding(
                id="r1-1",
                severity="bug",
                file="src/api.py",
                line=42,
                title="Missing auth check",
                evidence="Unauthenticated POST is accepted",
                status="open",
                models=("x-ai/grok-4.6",),
            ),
            _nit("r1-2"),
            _nit("r1-3"),
        ),
        reviewed_sha="b" * 40,
        generation="1234567890ab",
    )
    github = _LoopGitHub(encode_ledger(prior, repo=repo, pr_number=1))
    monkeypatch.setattr(cli_mod, "_collect", lambda env: _mk_collected())
    monkeypatch.setattr(cli_mod, "_github", lambda env: github)
    monkeypatch.setattr(cli_mod, "_maybe_status", lambda *args, **kwargs: None)

    lane_dir = tmp_path / "lanes"
    lane_dir.mkdir()
    (lane_dir / "lane-0.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "ok": True,
                "model": "x-ai/grok-4.6",
                "findings": [],
                "error": None,
                "head_sha": "a" * 40,
                # The floor removes the nits from the resolution contract, so
                # the lane owes a resolution only for the carried bug.
                "resolutions": [{"id": "r1-1", "status": "fixed", "note": "check added"}],
            }
        ),
        encoding="utf-8",
    )
    assert main(["judge"], _verify_env(tmp_path, lane_dir, repo)) == 0
    body = github.posted[0]
    assert "2 nit finding(s) from earlier rounds retired" in body
    assert "severity floor" in body
    updated = extract_ledger(body, repo=repo, pr_number=1)
    assert updated is not None
    assert updated.findings == ()  # bug fixed, nits retired — loop converges
    out = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert "verdict=clean" in out
    assert "issue_count=0" in out


def test_initial_round_embeds_marker_and_inline_comments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.collect import CollectedReview, DiffPlan, Truncation
    from or_pr_review.loop import extract_ledger

    repo = "FlyOverCoderKY/openrouter-pr-review-action"
    diff = (
        "diff --git a/src/api.py b/src/api.py\n"
        "--- a/src/api.py\n"
        "+++ b/src/api.py\n"
        "@@ -40,3 +40,4 @@\n"
        " ctx40\n"
        " ctx41\n"
        "+added42\n"
        " ctx43\n"
    )
    collected = CollectedReview(
        pr_number=1,
        title="t",
        body="",
        head_sha="a" * 40,
        base_ref="main",
        head_ref="feat",
        plan=DiffPlan("full-pr", "full-pr", None, "a" * 40, None),
        truncation=Truncation(diff, False, len(diff), len(diff), 300),
        mode="initial",
    )
    github = _LoopGitHub("unused")
    monkeypatch.setattr(cli_mod, "_collect", lambda env: collected)
    monkeypatch.setattr(cli_mod, "_github", lambda env: github)
    monkeypatch.setattr(cli_mod, "_maybe_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_mod, "_new_generation", lambda: "a" * 12)

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
                "head_sha": "a" * 40,
            }
        ),
        encoding="utf-8",
    )
    env = _base_env(
        tmp_path,
        LANE_RESULTS_DIR=str(lane_dir),
        PR_NUMBER="1",
        GITHUB_TOKEN="ghs_dummy",
        GITHUB_REPOSITORY=repo,
    )
    assert main(["judge"], env) == 0
    body = github.posted[0]
    updated = extract_ledger(body, repo=repo, pr_number=1)
    assert updated is not None
    assert updated.round_number == 1
    assert updated.findings[0].id == "r1-1"
    assert updated.findings[0].evidence.startswith("Unauthenticated POST")
    inline = github.comments[0]
    assert len(inline) == 1
    assert inline[0]["path"] == "src/api.py"
    assert inline[0]["line"] == 42
    # Marker is generation-scoped (initial round mints generation = sha[:12]).
    assert f"<!-- or-finding:{'a' * 12}:r1-1 -->" in inline[0]["body"]
    out = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert "round=1" in out
    assert "issue_count=1" in out


def test_force_push_resets_to_full_pr_initial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.collect import (
        DIVERGED_NOTICE,
        CollectedReview,
        DiffPlan,
        Truncation,
    )

    repo = "FlyOverCoderKY/openrouter-pr-review-action"
    github = _LoopGitHub(_prior_ledger_marker(repo))
    collected_calls: list[dict[str, str]] = []

    def fake_collect(env: dict[str, str]) -> CollectedReview:
        collected_calls.append(dict(env))
        if env.get("REVIEW_MODE") == "verify":
            plan = DiffPlan("latest-commit", "single-commit", None, "a" * 40, DIVERGED_NOTICE)
            return CollectedReview(
                1,
                "t",
                "",
                "a" * 40,
                "main",
                "feat",
                plan,
                Truncation("diff", False, 4, 4, 300),
                "verify",
            )
        plan = DiffPlan("full-pr", "full-pr", None, "a" * 40, None)
        return CollectedReview(
            1,
            "t",
            "",
            "a" * 40,
            "main",
            "feat",
            plan,
            Truncation("diff", False, 4, 4, 300),
            "initial",
        )

    monkeypatch.setattr(cli_mod, "_collect", fake_collect)
    monkeypatch.setattr(cli_mod, "_github", lambda env: github)
    env = _base_env(
        tmp_path,
        PR_NUMBER="1",
        GITHUB_TOKEN="ghs_dummy",
        GITHUB_REPOSITORY=repo,
        REVIEW_MODE="verify",
        REVIEW_SCOPE="latest-commit",
    )
    collected, state, replies = cli_mod._collect_with_loop(env)
    # The diverged range must not livelock in partial verify rounds: a
    # rewrite resets to a fresh full-PR initial round.
    assert state.mode == "initial"
    assert state.round_number == 1
    assert collected.plan.kind == "full-pr"
    assert replies == ""
    assert collected_calls[0]["EVENT_BEFORE"] == "b" * 40  # continuity attempted
    assert collected_calls[1]["REVIEW_MODE"] == "initial"
    assert collected_calls[1]["REVIEW_SCOPE"] == "full-pr"


def test_transient_compare_failure_does_not_reset_the_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.collect import (
        COMPARE_FAILED_NOTICE,
        CollectedReview,
        DiffPlan,
        Truncation,
    )

    repo = "FlyOverCoderKY/openrouter-pr-review-action"
    github = _LoopGitHub(_prior_ledger_marker(repo))
    collect_count = {"n": 0}

    def fake_collect(env: dict[str, str]) -> CollectedReview:
        collect_count["n"] += 1
        plan = DiffPlan("latest-commit", "single-commit", None, "a" * 40, COMPARE_FAILED_NOTICE)
        return CollectedReview(
            1,
            "t",
            "",
            "a" * 40,
            "main",
            "feat",
            plan,
            Truncation("diff", False, 4, 4, 300),
            "verify",
        )

    monkeypatch.setattr(cli_mod, "_collect", fake_collect)
    monkeypatch.setattr(cli_mod, "_github", lambda env: github)
    env = _base_env(
        tmp_path,
        PR_NUMBER="1",
        GITHUB_TOKEN="ghs_dummy",
        GITHUB_REPOSITORY=repo,
        REVIEW_MODE="verify",
        REVIEW_SCOPE="latest-commit",
    )
    collected, state, _replies = cli_mod._collect_with_loop(env)
    # A transient gh failure (timeout/5xx) must never wipe carried loop
    # state: the run stays a verify round with the fallback notice, and the
    # partial verdict downstream preserves the previous ledger.
    assert state.mode == "verify"
    assert state.round_number == 2
    assert state.prior_findings  # carried findings intact
    assert collected.plan.fallback_notice == COMPARE_FAILED_NOTICE
    assert collect_count["n"] == 1  # no re-collect / reset


def test_coverage_enforcement_skips_when_diff_exceeds_manifest_cap() -> None:
    from or_pr_review import cli as cli_mod
    from or_pr_review.collect import CollectedReview, DiffPlan, Truncation
    from or_pr_review.loop import LoopState
    from or_pr_review.schema import MAX_COVERAGE_ENTRIES

    diff = "".join(f"diff --git a/f{n}.txt b/f{n}.txt\n" for n in range(MAX_COVERAGE_ENTRIES + 1))
    collected = CollectedReview(
        1,
        "t",
        "",
        "a" * 40,
        "main",
        "feat",
        DiffPlan("full-pr", "full-pr", None, "a" * 40, None),
        Truncation(diff, False, len(diff), len(diff), 300),
        "initial",
    )
    state = LoopState(mode="initial", round_number=1)
    expect_coverage, expected_paths = cli_mod._coverage_expectations(state, collected)
    assert expect_coverage is False
    assert expected_paths is None
    # A normal-sized diff keeps enforcement on.
    small = CollectedReview(
        1,
        "t",
        "",
        "a" * 40,
        "main",
        "feat",
        DiffPlan("full-pr", "full-pr", None, "a" * 40, None),
        Truncation("diff --git a/x.py b/x.py\n", False, 4, 4, 300),
        "initial",
    )
    expect_coverage, expected_paths = cli_mod._coverage_expectations(state, small)
    assert expect_coverage is True
    assert expected_paths == {"x.py"}


def test_mixed_lane_commits_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from or_pr_review import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_collect", lambda env: _mk_collected())
    monkeypatch.setattr(cli_mod, "_maybe_status", lambda *args, **kwargs: None)

    lane_dir = tmp_path / "lanes"
    lane_dir.mkdir()
    for index, (model, sha) in enumerate(
        [("x-ai/grok-4.6", "a" * 40), ("anthropic/claude-sonnet-4.6", "b" * 40)]
    ):
        (lane_dir / f"lane-{index}.json").write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "ok": True,
                    "model": model,
                    "findings": [],
                    "error": None,
                    "head_sha": sha,
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
    )
    assert main(["judge"], env) == 1


def test_setup_rejects_bad_path_profiles(tmp_path: Path) -> None:
    env = _base_env(
        tmp_path,
        MODELS="x-ai/grok-4.6",
        PATH_PROFILES='[{"instructions": "no paths"}]',
    )
    assert main(["setup"], env) == 1


def test_stubbed_files_with_tools_disabled_stay_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace as dc_replace

    from or_pr_review import cli as cli_mod

    posted: list[str] = []

    class DummyGitHub:
        def create_review(self, number: int, body: str, commit_id: str) -> dict[str, object]:
            posted.append(body)
            return {"html_url": "https://example.test/review"}

        def pr_view(self, number: int) -> dict[str, object]:
            return {"headRefOid": "a" * 40}

    def collected_with_stubs(env: dict[str, str]):
        base = _mk_collected()
        return dc_replace(
            base,
            truncation=dc_replace(
                base.truncation,
                truncated=True,
                stubbed_files=("big.json",),
            ),
        )

    monkeypatch.setattr(cli_mod, "_collect", collected_with_stubs)
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
                "findings": [],
                "error": None,
                "head_sha": "a" * 40,
            }
        ),
        encoding="utf-8",
    )
    common = dict(
        LANE_RESULTS_DIR=str(lane_dir),
        PR_NUMBER="1",
        GITHUB_TOKEN="ghs_dummy",
        GITHUB_REPOSITORY="FlyOverCoderKY/openrouter-pr-review-action",
    )
    # Tools disabled: the stub contract cannot be honored -> partial.
    env = _base_env(tmp_path, MAX_TOOL_TURNS="0", **common)
    assert main(["judge"], env) == 0
    out = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert "verdict=partial" in out
    assert "could not be swept" in posted[0]

    # Tools available: stub-only truncation keeps the real verdict.
    (tmp_path / "out.txt").write_text("", encoding="utf-8")
    env = _base_env(tmp_path, MAX_TOOL_TURNS="50", **common)
    assert main(["judge"], env) == 0
    out = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert "verdict=clean" in out


def test_gitattributes_text_reads_the_reviewed_commit(tmp_path: Path) -> None:
    import subprocess

    from or_pr_review import cli as cli_mod

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            [
                "git",
                "-c",
                "user.email=t@example.invalid",
                "-c",
                "user.name=T",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    git("init", "-q", "-b", "main")
    (repo / ".gitattributes").write_text("*.json linguist-generated\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "c")
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    # Later working-tree edits must not leak: the read pins to the commit.
    (repo / ".gitattributes").write_text("* linguist-generated\n", encoding="utf-8")

    env = {"SOURCE_WORKSPACE": str(repo), "HEAD_SHA": sha}
    assert cli_mod._gitattributes_text(env) == "*.json linguist-generated\n"
    # A checkout without the reviewed commit (e.g. the judge job's action
    # checkout) fails soft to heuristics-only packing.
    env = {"SOURCE_WORKSPACE": str(repo), "HEAD_SHA": "b" * 40}
    assert cli_mod._gitattributes_text(env) == ""
    assert cli_mod._gitattributes_text({"SOURCE_WORKSPACE": str(repo)}) == ""
