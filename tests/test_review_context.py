from __future__ import annotations

import json
from dataclasses import replace

import pytest

from or_pr_review import cli, review_context
from or_pr_review.collect import CollectedReview, DiffPlan, Truncation
from or_pr_review.errors import SchemaError
from or_pr_review.loop import LedgerFinding, LoopState, extract_ledger
from or_pr_review.review_context import freeze_context, restore_context
from or_pr_review.schema import LaneResult, Resolution

REPO = "example/project"
HEAD = "a" * 40
DIFF = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n"


def collected():
    return CollectedReview(
        1,
        "Original title",
        "Original instructions",
        HEAD,
        "main",
        "change",
        DiffPlan("full-pr", "full-pr", "b" * 40, HEAD, None),
        Truncation(DIFF, False, len(DIFF), len(DIFF), 300),
        "verify",
        ("a.py",),
    )


def state():
    return LoopState(
        "verify",
        2,
        (
            LedgerFinding(
                "r1-1", "bug", "a.py", 1, "Prior bug", "Concrete evidence", "open", ("x/model",)
            ),
        ),
        "1234567890ab",
    )


def save_lane(directory, index=0, model="x/model", context=None):
    lane = LaneResult(
        1,
        True,
        model,
        head_sha=HEAD,
        resolutions=[Resolution("r1-1", "fixed", "Evidence of fix")],
        review_context=context or freeze_context(REPO, collected(), state(), 50),
    )
    (directory / f"lane-{index}.json").write_text(json.dumps(lane.to_dict()), encoding="utf-8")


def env(directory):
    return {
        "MODELS": "x/model",
        "GITHUB_REPOSITORY": REPO,
        "PR_NUMBER": "1",
        "MAX_TOOL_TURNS": "50",
        "LANE_RESULTS_DIR": str(directory),
        "STATUS_COMMENTS": "false",
        "GITHUB_OUTPUT": str(directory / "outputs"),
    }


@pytest.mark.parametrize("live_head", [HEAD, "c" * 40, None])
def test_matrix_uses_saved_state_without_recollection(tmp_path, monkeypatch, live_head):
    save_lane(tmp_path)
    posted = []

    class GitHub:
        def pr_view(self, number):
            return {"headRefOid": live_head}

        def create_review(self, number, body, commit, **kwargs):
            posted.append((body, commit, kwargs))
            return {}

    monkeypatch.setattr(cli, "_github", lambda _: GitHub())
    monkeypatch.setattr(
        cli, "_collect_with_loop", lambda *a, **k: pytest.fail("must not recollect")
    )
    assert cli.main(["judge"], env(tmp_path)) == 0
    body, head, kwargs = posted[0]
    assert head == HEAD
    assert "Round 2 resolution" in body
    ledger = extract_ledger(body, repo=REPO, pr_number=1)
    if live_head != HEAD:
        assert "pinned to commit" in body
        assert ledger is None
        assert not kwargs.get("comments")
        assert "verdict=partial" in (tmp_path / "outputs").read_text()
    else:
        assert ledger.generation == state().generation
        assert ledger.findings == ()
        assert ledger.round_number == 2


def test_all_role_artifacts_can_be_recovered_by_judge(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(cli, "_collect_with_loop", lambda _: (collected(), state(), ""))
    monkeypatch.setattr(cli, "_prepare_workspace", lambda *a: None)
    monkeypatch.setattr(cli, "_messages", lambda *a: [])
    monkeypatch.setattr(cli, "_maybe_status", lambda *a: None)
    monkeypatch.setattr(
        cli, "_invoke_lane", lambda _env, model, *a, **k: LaneResult(1, True, model)
    )

    def finish(_env, lanes, **kwargs):
        captured.append((lanes, kwargs))
        return 0

    monkeypatch.setattr(cli, "_finish", finish)
    settings = {**env(tmp_path), "MODELS": "x/model,y/model", "ALL_LANE_RESULTS_DIR": str(tmp_path)}
    assert cli.main(["all"], settings) == 0
    monkeypatch.setattr(
        cli, "_collect_with_loop", lambda *a, **k: pytest.fail("no fresh collection")
    )
    assert cli.main(["judge"], settings) == 0
    assert captured[0][1] == captured[1][1] == {"collected": collected(), "loop": state()}


@pytest.mark.parametrize("generation", ["", "a", "a" * 11, "a" * 12])
def test_snapshot_preserves_ledger_v1_generation_compatibility(generation):
    original = replace(state(), generation=generation)
    assert restore_context(freeze_context(REPO, collected(), original, 50)).loop == original


def test_invalid_unicode_is_a_schema_failure():
    invalid = replace(collected(), truncation=Truncation("\ud800", False, 1, 1, 300))
    with pytest.raises(SchemaError, match="UTF-8"):
        freeze_context(REPO, invalid, state(), 50)


def test_empty_matrix_reports_incomplete_without_recollection(tmp_path, monkeypatch):
    notices = []
    monkeypatch.setattr(
        cli, "_collect_with_loop", lambda *a, **k: pytest.fail("no context available")
    )
    monkeypatch.setattr(cli, "_best_effort_incomplete", lambda _env, **kw: notices.append(kw))
    assert cli.main(["judge"], env(tmp_path)) == 1
    assert notices[0]["stage"] == "judge"
    assert "no matrix publication context" in notices[0]["reason"]


def test_same_head_different_ledger_rejected_before_judge(tmp_path, monkeypatch):
    save_lane(tmp_path)
    save_lane(
        tmp_path,
        1,
        "y/model",
        freeze_context(REPO, collected(), replace(state(), generation="b" * 12), 50),
    )
    monkeypatch.setattr(
        cli, "_finish", lambda *a, **k: pytest.fail("must reject before publication")
    )
    with pytest.raises(SchemaError, match="different review contexts"):
        cli._role_judge({**env(tmp_path), "MODELS": "x/model,y/model"})


@pytest.mark.parametrize(
    "field,value",
    [
        ("GITHUB_REPOSITORY", "another/repo"),
        ("PR_NUMBER", "2"),
        ("HEAD_SHA", "c" * 40),
        ("MAX_TOOL_TURNS", "0"),
    ],
)
def test_judge_identity_mismatch_fails_before_publication(tmp_path, monkeypatch, field, value):
    save_lane(tmp_path)
    monkeypatch.setattr(cli, "_finish", lambda *a, **k: pytest.fail("must not publish"))
    with pytest.raises(SchemaError, match="does not match"):
        cli._role_judge({**env(tmp_path), field: value})


@pytest.mark.parametrize("mutation", ["missing", "digest", "head", "model"])
def test_invalid_artifact_rejected(tmp_path, mutation):
    save_lane(tmp_path)
    path = tmp_path / "lane-0.json"
    payload = json.loads(path.read_text())
    if mutation == "missing":
        del payload["review_context"]
    elif mutation == "digest":
        payload["review_context"]["payload"]["collected"]["body"] = "changed"
    else:
        payload["head_sha" if mutation == "head" else "model"] = "c" * 40
    path.write_text(json.dumps(payload))
    with pytest.raises(SchemaError):
        cli._load_lane_dir(tmp_path, ["x/model"])


def test_missing_sibling_keeps_completed_lane_context(tmp_path, monkeypatch):
    save_lane(tmp_path)
    seen = {}

    def finish(_env, lanes, **kwargs):
        seen.update(kwargs)
        assert lanes[0].ok and not lanes[1].ok
        return 0

    monkeypatch.setattr(cli, "_finish", finish)
    assert cli._role_judge({**env(tmp_path), "MODELS": "x/model,y/model"}) == 0
    assert seen == {"collected": collected(), "loop": state()}


def test_context_roundtrip_and_type_validation():
    snapshot = freeze_context(REPO, collected(), state(), 50)
    context = restore_context(json.loads(json.dumps(snapshot)))
    assert context.collected == collected()
    assert context.loop == state()
    with pytest.raises(SchemaError):
        freeze_context(REPO, replace(collected(), pr_number=True), state(), 50)


def test_context_size_checked_before_lane_spend(tmp_path, monkeypatch):
    monkeypatch.setattr(review_context, "MAX_CONTEXT_BYTES", 10)
    monkeypatch.setattr(cli, "_collect_with_loop", lambda _: (collected(), state(), ""))
    monkeypatch.setattr(cli, "_invoke_lane", lambda *a, **k: pytest.fail("must not spend"))
    with pytest.raises(SchemaError, match="artifact limit"):
        cli._run_one_lane(env(tmp_path), "x/model")
