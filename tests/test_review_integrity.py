"""Offline regressions for source preservation, completeness and paid usage."""

import json
from pathlib import Path

import pytest

from or_pr_review import cli
from or_pr_review.collect import CollectedReview, DiffPlan, Truncation
from or_pr_review.errors import LaneError, SchemaError
from or_pr_review.harness import run_lane
from or_pr_review.judge import run_llm_judge
from or_pr_review.loop import LoopState
from or_pr_review.publish import _cost_note
from or_pr_review.schema import MAX_FINDINGS, LaneResult, parse_lane_artifact, parse_lane_payload


def finding(title="Missing boundary check", body="An empty list raises instead of returning zero."):
    return dict(title=title, body=body, severity="bug", file="a.py", line=10)


def response(payload, cost=0.03):
    return {"choices": [{"message": {"content": json.dumps(payload)}}], "usage": {"cost": cost}}


def test_judge_cannot_rewrite_source_evidence_or_metadata():
    source = finding()
    lanes = [dict(model="example/a", findings=[source]), dict(model="example/b", findings=[])]
    invented = dict(
        title="Everything works",
        body="No failure is possible.",
        severity="nit",
        file="other.py",
        line=99,
        models=["example/b"],
        sources=["0.0"],
    )
    issues, mode, _ = run_llm_judge(
        model="example/judge",
        lanes=lanes,
        api_key="unused",
        chat=lambda _: response({"issues": [invented]}),
    )
    issue = issues[0]
    assert (issue.title, issue.body, issue.severity, issue.file, issue.line) == (
        source["title"],
        source["body"],
        "bug",
        "a.py",
        10,
    )
    assert issue.models == ["example/a"]
    assert "canonical" in mode


def test_nearby_distinct_defects_survive_a_fully_accounted_lump():
    a = finding()
    b = {
        **finding("Wrong rounding", "Fractional values are truncated before summation."),
        "line": 12,
    }
    merged = {**a, "models": ["example/a"], "sources": ["0.0", "0.1"]}
    issues, mode, _ = run_llm_judge(
        model="example/judge",
        lanes=[dict(model="example/a", findings=[a, b])],
        api_key="unused",
        chat=lambda _: response({"issues": [merged]}),
    )
    assert {(i.title, i.body, i.line) for i in issues} == {
        (x["title"], x["body"], x["line"]) for x in (a, b)
    }
    assert "split+2" in mode


def test_lane_overflow_retains_late_bug_and_roundtrips_omissions():
    nits = [{**finding(str(i)), "severity": "nit"} for i in range(MAX_FINDINGS)]
    result = run_lane(
        model="example/a",
        messages=[],
        api_key="unused",
        workspace=None,
        max_tool_turns=0,
        chat=lambda _: response({"findings": [*nits, finding()]}),
    )
    assert result.ok
    assert len(result.findings) == MAX_FINDINGS
    assert result.findings[0].severity == "bug"
    assert parse_lane_artifact(result.to_dict()).dropped_findings == 1
    with pytest.raises(LaneError):
        parse_lane_payload({"findings": [*nits, {"severity": "bug"}]}, "example/a")


@pytest.mark.parametrize("invalid", [-1, True, None, "1"])
def test_invalid_omission_metadata_fails_closed(invalid):
    payload = LaneResult(1, True, "example/a").to_dict()
    payload["dropped_findings"] = invalid
    with pytest.raises(SchemaError, match="dropped_findings"):
        parse_lane_artifact(payload)


def test_omissions_publish_partial_without_an_authoritative_ledger(monkeypatch, tmp_path):
    posted = []
    outputs = {}

    class GitHub:
        def pr_view(self, _number):
            return {"headRefOid": "a" * 40}

        def create_review(self, _number, body, _sha):
            posted.append(body)
            return {}

    monkeypatch.setattr(cli, "_github", lambda _: GitHub())
    monkeypatch.setattr(cli, "_set_output", lambda key, value: outputs.update({key: value}))
    monkeypatch.setattr(cli, "_maybe_status", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_checkout_has_commit", lambda *a: False)
    collected = CollectedReview(
        1,
        "Example",
        "",
        "a" * 40,
        "main",
        "branch",
        DiffPlan("full-pr", "full-pr", None, "a" * 40, None),
        Truncation("", False, 0, 0, 300),
        "initial",
    )
    env = {"MODELS": "example/a", "FAIL_ON": "never", "SOURCE_WORKSPACE": str(tmp_path)}
    lane = LaneResult(1, True, "example/a", head_sha="a" * 40, dropped_findings=1)
    assert cli._finish(env, [lane], collected, LoopState("initial", 1)) == 0
    assert outputs["verdict"] == "partial"
    assert "omitted 1 finding(s)" in posted[0]
    assert "<!-- openrouter-review-ledger:" not in posted[0]


@pytest.mark.parametrize("payload", [{"issues": "bad"}, {"choices": []}])
def test_unusable_judge_response_retains_cost_through_publication(monkeypatch, payload):
    def fake_judge(**kwargs):
        reply = response(payload) if "issues" in payload else {**payload, "usage": {"cost": 0.03}}
        return run_llm_judge(**kwargs, chat=lambda _: reply)

    monkeypatch.setattr(cli, "run_llm_judge", fake_judge)
    lanes = [LaneResult(1, True, f"example/{x}", cost_usd=0.1) for x in ("a", "b")]
    outcome = cli._resolve_issues(
        {"OPENROUTER_API_KEY": "unused", "JUDGE_MODEL": "example/judge"},
        [lane.model for lane in lanes],
        lanes,
        lanes,
    )
    assert outcome.ran
    assert outcome.cost == pytest.approx(0.03)
    assert "$0.2300" in _cost_note(lanes, outcome.cost, outcome.ran)
    assert "incomplete" in _cost_note(lanes, None, True)


def test_matrix_judge_receives_the_lane_tool_policy():
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/pr-review.yml").read_text()
    judge = workflow.split("\n  judge:", 1)[1]
    assert "max_tool_turns: ${{ inputs.max_tool_turns }}" in judge
