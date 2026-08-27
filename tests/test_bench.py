from __future__ import annotations

import json
from pathlib import Path

import pytest

from or_pr_review.bench import (
    Label,
    collected_from_fixture,
    load_fixture,
    main,
    match_finding,
    score_run,
)
from or_pr_review.errors import ActionError
from or_pr_review.prompt import build_messages, changed_paths_from_diff

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "bench" / "fixtures" / "planted-mini"


def test_planted_fixture_loads_and_builds_messages() -> None:
    fixture = load_fixture(FIXTURE_DIR)
    assert len(fixture.labels) == 11
    severities = [label.severity for label in fixture.labels]
    assert severities.count("bug") == 2
    assert severities.count("risk") == 4
    assert severities.count("nit") == 5
    assert "diff --git a/calc.py" in fixture.diff
    # The blast-radius plant: docs/rules.md is in the checkout but NOT the diff.
    assert (fixture.checkout / "docs" / "rules.md").is_file()
    assert "docs/rules.md" not in changed_paths_from_diff(fixture.diff)
    # The fixture must NOT hint at its own plants through custom instructions.
    assert fixture.custom_instructions == ""
    collected = collected_from_fixture(fixture)
    messages = build_messages(collected, custom_instructions=fixture.custom_instructions)
    assert messages[0]["role"] == "system"
    assert "prefer recall over precision" in messages[0]["content"]
    assert "CAP_2027" in messages[1]["content"]


def test_match_finding_requires_file_and_keyword() -> None:
    label = Label(
        id="B1", severity="bug", file="calc.py", title="t", keywords=("KeyError",)
    )
    hit = {"file": "calc.py", "title": "apply_cap raises KeyError", "body": ""}
    wrong_file = {"file": "rules.py", "title": "apply_cap raises KeyError", "body": ""}
    wrong_text = {"file": "calc.py", "title": "something else", "body": "no match"}
    subdir = {"file": "src/calc.py", "title": "KeyError on other years", "body": ""}
    assert match_finding(hit, label)
    assert not match_finding(wrong_file, label)
    assert not match_finding(wrong_text, label)
    assert match_finding(subdir, label)  # suffix path match
    fileless = Label(id="X", severity="nit", file=None, title="t", keywords=("spelling",))
    assert match_finding({"file": None, "title": "Spelling drift", "body": ""}, fileless)


def test_score_run_recall_precision_and_unmatched() -> None:
    labels = (
        Label(id="B1", severity="bug", file="calc.py", title="t", keywords=("KeyError",)),
        Label(id="N1", severity="nit", file="rules.py", title="t", keywords=("duplicate",)),
    )
    findings = [
        {"file": "calc.py", "title": "KeyError for 2025", "body": "", "severity": "bug"},
        {"file": "docs/x.md", "title": "unrelated noise", "body": "", "severity": "nit"},
    ]
    score = score_run(findings, labels)
    assert score.recall(labels) == (1, 2)
    assert score.recall(labels, "bug") == (1, 1)
    assert score.recall(labels, "nit") == (0, 1)
    assert score.precision() == (1, 2)
    assert [f["title"] for f in score.unmatched_findings] == ["unrelated noise"]


def test_precision_counts_findings_not_distinct_titles() -> None:
    labels = (
        Label(id="A", severity="bug", file="a.py", title="t", keywords=("missing validation",)),
        Label(id="B", severity="bug", file="b.py", title="t", keywords=("missing validation",)),
    )
    findings = [
        {"file": "a.py", "title": "Missing validation", "body": "", "severity": "bug"},
        {"file": "b.py", "title": "Missing validation", "body": "", "severity": "bug"},
    ]
    score = score_run(findings, labels)
    assert score.precision() == (2, 2)  # duplicate titles must both count
    assert score.recall(labels) == (2, 2)


def test_severity_agreement_tracks_matched_severities() -> None:
    labels = (
        Label(id="B1", severity="bug", file="calc.py", title="t", keywords=("KeyError",)),
        Label(id="N1", severity="nit", file="rules.py", title="t", keywords=("duplicate",)),
    )
    findings = [
        # Detected, but reported at the wrong severity.
        {"file": "calc.py", "title": "KeyError edge", "body": "", "severity": "nit"},
        {"file": "rules.py", "title": "duplicate quote", "body": "", "severity": "nit"},
    ]
    score = score_run(findings, labels)
    assert score.recall(labels) == (2, 2)  # detection is severity-agnostic
    assert score.severity_agreement(labels) == (1, 2)


def _write_fixture(fixture_dir: Path, labels: object) -> None:
    (fixture_dir / "checkout").mkdir(parents=True, exist_ok=True)
    (fixture_dir / "diff.patch").write_text(
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n", encoding="utf-8"
    )
    (fixture_dir / "fixture.json").write_text(json.dumps({"title": "t"}), encoding="utf-8")
    (fixture_dir / "labels.json").write_text(json.dumps(labels), encoding="utf-8")


def test_label_validation_fails_fast(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "f"
    _write_fixture(fixture_dir, [{"id": "X", "severity": "bug", "keywords": ["("]}])
    with pytest.raises(ActionError, match="not a valid regex"):
        load_fixture(fixture_dir)
    _write_fixture(fixture_dir, [{"id": "X", "severity": "urgent", "keywords": ["ok"]}])
    with pytest.raises(ActionError, match="severity"):
        load_fixture(fixture_dir)
    # A bare string would explode into one regex per character.
    _write_fixture(fixture_dir, [{"id": "X", "severity": "bug", "keywords": "KeyError"}])
    with pytest.raises(ActionError, match="non-empty list"):
        load_fixture(fixture_dir)
    # An empty keyword matches every finding.
    _write_fixture(fixture_dir, [{"id": "X", "severity": "bug", "keywords": ["ok", "  "]}])
    with pytest.raises(ActionError, match="empty keyword"):
        load_fixture(fixture_dir)
    _write_fixture(fixture_dir, [{"severity": "bug", "keywords": ["ok"]}])
    with pytest.raises(ActionError, match="missing its id"):
        load_fixture(fixture_dir)


def test_fixture_paths_are_confined(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "f"
    _write_fixture(fixture_dir, [])
    meta = {"title": "t", "checkout_dir": "../outside"}
    (tmp_path / "outside").mkdir()
    (fixture_dir / "fixture.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ActionError, match="escapes the fixture directory"):
        load_fixture(fixture_dir)


def _fake_lane(ok: bool = True, findings: list | None = None):
    from or_pr_review.schema import SCHEMA_VERSION, LaneResult

    return LaneResult(
        schema_version=SCHEMA_VERSION,
        ok=ok,
        model="x-ai/grok-4.6",
        findings=findings or [],
        error=None if ok else "boom",
        tool_rounds=1,
    )


def test_cmd_run_clears_stale_files_and_flags_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import bench as bench_mod

    fixture_dir = tmp_path / "f"
    _write_fixture(fixture_dir, [])
    out = tmp_path / "out"
    out.mkdir()
    stale = out / "run-7.json"
    stale.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(bench_mod, "run_lane", lambda **kwargs: _fake_lane(ok=True))
    rc = main(["run", str(fixture_dir), "--runs", "2", "--out", str(out)])
    assert rc == 0
    assert not stale.exists()  # stale run files must not contaminate `score`
    assert sorted(p.name for p in out.glob("run-*.json")) == ["run-0.json", "run-1.json"]

    monkeypatch.setattr(bench_mod, "run_lane", lambda **kwargs: _fake_lane(ok=False))
    assert main(["run", str(fixture_dir), "--runs", "1", "--out", str(out)]) == 1

    assert main(["run", str(fixture_dir), "--runs", "0", "--out", str(out)]) == 1


def test_cmd_score_reports_means_and_skips_failed_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture_dir = tmp_path / "f"
    _write_fixture(
        fixture_dir,
        [{"id": "B1", "severity": "bug", "file": "a.py", "keywords": ["KeyError"]}],
    )
    out = tmp_path / "out"
    out.mkdir()
    good = {
        "ok": True,
        "findings": [
            {"file": "a.py", "title": "KeyError", "body": "", "severity": "bug"}
        ],
    }
    (out / "run-0.json").write_text(json.dumps(good), encoding="utf-8")
    (out / "run-3.json").write_text(json.dumps({"ok": False, "error": "x"}), encoding="utf-8")
    assert main(["score", str(fixture_dir), str(out)]) == 0
    printed = capsys.readouterr().out
    assert "1 scored run(s), 1 failed run(s)" in printed
    assert "run-0 | 1/1" in printed  # rows named by file, not enumeration
    assert "mean | 100%" in printed
