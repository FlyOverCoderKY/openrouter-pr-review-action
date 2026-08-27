from __future__ import annotations

from pathlib import Path

import pytest

from or_pr_review.bench import (
    Label,
    collected_from_fixture,
    load_fixture,
    match_finding,
    score_run,
)
from or_pr_review.errors import ActionError
from or_pr_review.prompt import build_messages, changed_paths_from_diff

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "bench" / "fixtures" / "planted-mini"


def test_planted_fixture_loads_and_builds_messages() -> None:
    fixture = load_fixture(FIXTURE_DIR)
    assert len(fixture.labels) == 10
    severities = [label.severity for label in fixture.labels]
    assert severities.count("bug") == 2
    assert severities.count("risk") == 4
    assert severities.count("nit") == 4
    assert "diff --git a/calc.py" in fixture.diff
    # The blast-radius plant: docs/rules.md is in the checkout but NOT the diff.
    assert (fixture.checkout / "docs" / "rules.md").is_file()
    assert "docs/rules.md" not in changed_paths_from_diff(fixture.diff)
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


def test_bad_label_regex_fails_fast(tmp_path: Path) -> None:
    import json

    fixture_dir = tmp_path / "f"
    (fixture_dir / "checkout").mkdir(parents=True)
    (fixture_dir / "diff.patch").write_text("", encoding="utf-8")
    (fixture_dir / "fixture.json").write_text(json.dumps({"title": "t"}), encoding="utf-8")
    (fixture_dir / "labels.json").write_text(
        json.dumps([{"id": "X", "severity": "bug", "keywords": ["("]}]), encoding="utf-8"
    )
    with pytest.raises(Exception):
        load_fixture(fixture_dir)
    (fixture_dir / "labels.json").write_text(
        json.dumps([{"id": "X", "severity": "urgent", "keywords": ["ok"]}]), encoding="utf-8"
    )
    with pytest.raises(ActionError, match="severity"):
        load_fixture(fixture_dir)
