from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from or_pr_review.bench import (
    Adjudication,
    JudgeFixture,
    Label,
    collected_from_fixture,
    load_fixture,
    load_judge_fixture,
    main,
    match_finding,
    score_judge_output,
    score_run,
)
from or_pr_review.errors import ActionError
from or_pr_review.prompt import build_messages, changed_paths_from_diff

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "bench" / "fixtures" / "planted-mini"
JUDGE_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "bench" / "judge-fixtures"


def _load_capture():
    path = FIXTURE_DIR.parent.parent / "capture.py"
    spec = importlib.util.spec_from_file_location("bench_capture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_planted_fixture_loads_and_builds_messages() -> None:
    fixture = load_fixture(FIXTURE_DIR)
    assert len(fixture.labels) == 13
    severities = [label.severity for label in fixture.labels]
    assert severities.count("bug") == 2
    assert severities.count("risk") == 6
    assert severities.count("nit") == 5
    assert "diff --git a/calc.py" in fixture.diff
    # The blast-radius plant: docs/rules.md is in the checkout but NOT the diff.
    assert (fixture.checkout / "docs" / "rules.md").is_file()
    assert "docs/rules.md" not in changed_paths_from_diff(fixture.diff)
    # Containment: the file/repo-context plants must stay OUT of the planted
    # diff (their strata claims depend on it) while the clean twin fixes them.
    assert "SUPPORTED_YEARS" not in fixture.diff
    assert "report.py" not in fixture.diff
    assert (fixture.checkout / "report.py").is_file()
    clean = load_fixture(FIXTURE_DIR.parent / "planted-mini-clean")
    assert "SUPPORTED_YEARS" in clean.diff
    assert "report.py" in clean.diff
    # Semantic containment: the file/repo labels must be UNMATCHABLE from the
    # diff alone — including via hunk headers — and must not cross-credit
    # correct diff-stratum findings (the exact vectors the self-review found).
    diff_finding = {
        "file": "calc.py",
        "title": "quotes the diff",
        "body": fixture.diff,
        "severity": "risk",
    }
    for lid in ("F1", "R6"):
        label = next(item for item in fixture.labels if item.id == lid)
        assert not match_finding(diff_finding, label), lid
        assert not match_finding({**diff_finding, "file": None}, label), lid
    f1 = next(item for item in fixture.labels if item.id == "F1")
    b2_confounder = {
        "file": "calc.py",
        "severity": "bug",
        "title": "apply_cap raises KeyError for unsupported years",
        "body": (
            "Year validation is missing or out of sync: caps has only 2026 and "
            "2027 and other years crash despite the docstring."
        ),
    }
    assert not match_finding(b2_confounder, f1)
    r6 = next(item for item in fixture.labels if item.id == "R6")
    default_confounder = {
        "file": "calc.py",
        "severity": "nit",
        "title": "apply_cap default year stays 2026",
        "body": (
            "Callers relying on the default get 2026 caps. Falsification: checked "
            "report.py, rules.py, tests."
        ),
    }
    assert not match_finding(default_confounder, r6)
    # Bare-symbol mentions with unrelated semantics must not credit either
    # context label (the Codex P1 vector).
    assert not match_finding(
        {
            "file": "report.py",
            "severity": "nit",
            "title": "annual_report returns a bare dict",
            "body": "A typed result object would be clearer than a dict.",
        },
        r6,
    )
    assert not match_finding(
        {
            "file": "calc.py",
            "severity": "nit",
            "title": "validate_year lacks a docstring",
            "body": "It raises ValueError for unsupported years but never documents that.",
        },
        f1,
    )
    assert match_finding(
        {
            "file": "calc.py",
            "severity": "risk",
            "title": "Year gate out of date",
            "body": (
                "SUPPORTED_YEARS still lists only 2026, so validate_year rejects "
                "2027 even though apply_cap now supports it."
            ),
        },
        f1,
    )
    genuine_r6 = {
        "file": "report.py",
        "severity": "risk",
        "title": "annual_report caps with the default year",
        "body": (
            "annual_report ignores its year argument when capping: apply_cap uses "
            "the default 2026 even for year=2027."
        ),
    }
    assert match_finding(genuine_r6, r6)
    # The fixture must not coach its own plants: no fourth-wall language.
    for banned in ("padding", "must require reading", "outside the diff"):
        assert banned not in (fixture.checkout / "calc.py").read_text(encoding="utf-8")
        assert banned not in clean.diff
    # The fixture must NOT hint at its own plants through custom instructions.
    assert fixture.custom_instructions == ""
    collected = collected_from_fixture(fixture)
    messages = build_messages(collected, custom_instructions=fixture.custom_instructions)
    assert messages[0]["role"] == "system"
    assert "prefer recall over precision" in messages[0]["content"]
    assert "CAP_2027" in messages[1]["content"]


def test_match_finding_requires_file_and_keyword() -> None:
    label = Label(id="B1", severity="bug", file="calc.py", title="t", keywords=("KeyError",))
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
    # Without an adjudication the unmatched finding is a third class, not
    # auto-false: precision counts only adjudicated outcomes.
    assert score.precision() == (1, 1)
    assert [f["title"] for f in score.unadjudicated] == ["unrelated noise"]


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


def test_context_strata_and_adjudication_classification() -> None:
    labels = (
        Label(id="D1", severity="bug", file="a.py", title="t", keywords=("KeyError",)),
        Label(
            id="R9",
            severity="risk",
            file=None,
            title="t",
            keywords=("inventory",),
            context="repo",
        ),
    )
    adjudications = (
        Adjudication(
            id="A1",
            verdict="true_positive_unlabeled",
            file=None,
            keywords=("missing docstring",),
        ),
        Adjudication(id="A2", verdict="false_positive", file=None, keywords=("cosmic ray",)),
    )
    findings = [
        {"file": "a.py", "title": "KeyError on 2025", "body": "", "severity": "bug"},
        {"file": "b.py", "title": "Missing docstring", "body": "", "severity": "nit"},
        {"file": "b.py", "title": "cosmic ray hazard", "body": "", "severity": "risk"},
        {"file": "c.py", "title": "mystery finding", "body": "", "severity": "nit"},
    ]
    score = score_run(findings, labels, adjudications)
    assert score.recall(labels, context="diff") == (1, 1)
    assert score.recall(labels, context="repo") == (0, 1)
    assert [(f["title"], aid) for f, aid in score.adjudicated_tp] == [("Missing docstring", "A1")]
    assert [(f["title"], aid) for f, aid in score.adjudicated_fp] == [("cosmic ray hazard", "A2")]
    assert [f["title"] for f in score.unadjudicated] == ["mystery finding"]
    # precision: (1 label match + 1 adjudicated TP) over those plus 1 FP;
    # the unadjudicated finding is a third class, not auto-false.
    assert score.precision() == (2, 3)


def test_planted_fixture_context_labels_and_adjudications() -> None:
    fixture = load_fixture(FIXTURE_DIR)
    contexts = {label.id: label.context for label in fixture.labels}
    assert contexts["R4"] == "repo"  # the docs-inventory plant needs tool use
    assert contexts["R6"] == "repo"  # the report.py caller plant lives outside the diff
    assert contexts["F1"] == "file"  # SUPPORTED_YEARS is only visible by reading calc.py
    assert all(c == "diff" for lid, c in contexts.items() if lid not in {"R4", "R6", "F1"})
    assert any(a.verdict == "true_positive_unlabeled" for a in fixture.adjudications)


def test_clean_twin_fixture_loads_with_zero_labels() -> None:
    clean_dir = FIXTURE_DIR.parent / "planted-mini-clean"
    fixture = load_fixture(clean_dir)
    assert fixture.labels == ()
    assert "diff --git a/calc.py" in fixture.diff
    # The clean twin fixes the plants: sourced 2027 dollar, year guard,
    # kept validation, docs row present, consistent id spelling.
    assert "8_550" in fixture.diff
    rules = (fixture.checkout / "docs" / "rules.md").read_text(encoding="utf-8")
    assert rules.count("hsa-cap-") >= 2
    assert "modelled" not in fixture.diff
    # Scoring a clean fixture: every finding is a noise candidate.
    score = score_run(
        [{"file": "calc.py", "title": "anything", "body": "", "severity": "nit"}],
        fixture.labels,
        fixture.adjudications,
    )
    assert score.precision() == (0, 0)
    assert len(score.unadjudicated) == 1


def test_committed_fixtures_pin_the_production_embed_cap() -> None:
    from or_pr_review.bench import DEFAULT_MAX_DIFF_KB

    assert load_fixture(FIXTURE_DIR).max_diff_kb == DEFAULT_MAX_DIFF_KB
    assert (
        load_fixture(FIXTURE_DIR.parent / "planted-mini-clean").max_diff_kb == DEFAULT_MAX_DIFF_KB
    )


def test_capture_default_matches_production_embed_cap() -> None:
    from or_pr_review.collect import DEFAULT_MAX_DIFF_KB

    assert _load_capture().DEFAULT_MAX_DIFF_KB == DEFAULT_MAX_DIFF_KB


def test_bench_run_cli_defaults_follow_production_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from or_pr_review import bench as bench_mod

    seen = {}

    def capture(args: object) -> int:
        seen.update(vars(args))
        return 0

    monkeypatch.setattr(bench_mod, "_cmd_run", capture)
    assert bench_mod.main(["run", "fixture", "--out", "out"]) == 0
    assert seen["model"] == bench_mod.DEFAULT_MODEL
    assert seen["max_tool_turns"] == bench_mod.DEFAULT_MAX_TOOL_TURNS
    assert seen["timeout"] == bench_mod.DEFAULT_TIMEOUT


def test_capture_modern_extraction_does_not_mask_type_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _load_capture()
    monkeypatch.setattr(capture.tarfile, "data_filter", object(), raising=False)
    extract_archive = capture._extract_archive

    checkout = tmp_path / "checkout"
    checkout.mkdir()

    class FakeTar:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple, dict]] = []

        def extractall(self, *args: object, **kwargs: object) -> None:
            self.calls.append((args, kwargs))
            raise TypeError("bug inside modern extraction")

        def getmembers(self) -> list[tarfile.TarInfo]:
            raise AssertionError("modern extraction must not enter the legacy path")

    tar = FakeTar()
    with pytest.raises(TypeError, match="bug inside modern extraction"):
        extract_archive(tar, checkout)
    assert tar.calls == [((checkout,), {"filter": "data"})]


def test_documented_capture_script_starts_without_install(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "bench" / "capture.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Capture a real PR" in result.stdout


def test_capture_legacy_extraction_validates_members_before_extracting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _load_capture()
    monkeypatch.delattr(capture.tarfile, "data_filter", raising=False)
    checkout = tmp_path / "checkout"
    checkout.mkdir()

    class FakeTar:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple, dict]] = []

        def extractall(self, *args: object, **kwargs: object) -> None:
            self.calls.append((args, kwargs))

        def getmembers(self) -> list[tarfile.TarInfo]:
            return [tarfile.TarInfo("safe.txt")]

    tar = FakeTar()
    capture._extract_archive(tar, checkout)
    assert tar.calls == [((checkout,), {})]


def test_capture_legacy_extraction_rejects_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _load_capture()
    monkeypatch.delattr(capture.tarfile, "data_filter", raising=False)

    checkout = tmp_path / "checkout"
    checkout.mkdir()

    class FakeTar:
        def extractall(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("unsafe archive must not be extracted")

        def getmembers(self) -> list[tarfile.TarInfo]:
            link = tarfile.TarInfo("link")
            link.type = tarfile.SYMTYPE
            link.linkname = "../../outside"
            return [link]

    with pytest.raises(SystemExit, match="links are not allowed"):
        capture._extract_archive(FakeTar(), checkout)


def _load_generator():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "generate_planted", FIXTURE_DIR.parent / "generate_planted.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_checkouts_match_the_generator() -> None:
    module = _load_generator()
    for name, head in module.FIXTURE_HEADS.items():
        checkout = FIXTURE_DIR.parent / name / "checkout"
        # Byte comparison (not text) so newline drift fails too, and the
        # committed file SET must equal the generator tree — extras fail.
        committed_files = sorted(
            str(p.relative_to(checkout)).replace("\\", "/")
            for p in checkout.rglob("*")
            if p.is_file()
        )
        assert committed_files == sorted(head), f"{name} checkout file set drifted"
        for rel, content in head.items():
            committed = (checkout / rel).read_bytes()
            assert committed == content.encode("utf-8"), (
                f"{name}/{rel} drifted from generate_planted.py"
            )


def test_committed_diffs_match_the_generator() -> None:
    # Regenerate each diff with the generator's isolated git and compare
    # byte-for-byte, so a stale diff.patch cannot disagree with the checkout.
    if shutil.which("git") is None:
        pytest.skip("git is required to regenerate committed fixture diffs")
    module = _load_generator()
    for name, head in module.FIXTURE_HEADS.items():
        committed = (FIXTURE_DIR.parent / name / "diff.patch").read_bytes()
        regenerated = module.render_diff(head).encode("utf-8")
        assert committed == regenerated, f"{name}/diff.patch drifted from the generator"


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
    # context is explicit, never defaulted — a missing or bogus value fails.
    _write_fixture(fixture_dir, [{"id": "X", "severity": "bug", "keywords": ["ok"]}])
    with pytest.raises(ActionError, match="context"):
        load_fixture(fixture_dir)
    _write_fixture(
        fixture_dir,
        [{"id": "X", "severity": "bug", "keywords": ["ok"], "context": "galaxy"}],
    )
    with pytest.raises(ActionError, match="context"):
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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("diff_file", 7, "diff_file must be a non-empty string path"),
        ("checkout_dir", [], "checkout_dir must be a non-empty string path"),
        ("labels_file", None, "labels_file must be a non-empty string path"),
        ("pr_number", "not-a-number", "pr_number must be a positive integer"),
        ("pr_number", 1.5, "pr_number must be a positive integer"),
        ("title", {"unexpected": "object"}, "title must be a string"),
    ],
)
def test_fixture_metadata_type_errors_are_action_errors(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    fixture_dir = tmp_path / "f"
    _write_fixture(fixture_dir, [])
    meta = {"title": "t", field: value}
    (fixture_dir / "fixture.json").write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ActionError, match=message):
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


def test_cmd_run_requires_allow_spend_before_other_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from or_pr_review import bench as bench_mod

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-present-but-not-consent")
    called = False

    def unexpected_load(path: Path):
        nonlocal called
        called = True
        raise AssertionError("fixture loading must not happen before spend consent")

    monkeypatch.setattr(bench_mod, "load_fixture", unexpected_load)
    out = tmp_path / "out"
    assert main(["run", "missing-fixture", "--out", str(out)]) == 1
    assert not called
    assert not out.exists()
    assert "re-run with --allow-spend" in capsys.readouterr().err


def test_cmd_run_requires_api_key_after_spend_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture_dir = tmp_path / "fixture"
    _write_fixture(fixture_dir, [])
    out = tmp_path / "out"
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert main(["run", str(fixture_dir), "--out", str(out), "--allow-spend"]) == 1
    assert not out.exists()
    assert "OPENROUTER_API_KEY is not set" in capsys.readouterr().err


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
    stale_progress = out / "progress-7.json"
    stale_progress.write_text("{}", encoding="utf-8")
    stale_progress_tmp = out / "progress-7.tmp"
    stale_progress_tmp.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    def successful(**kwargs):
        kwargs["progress"]({"elapsed_ms": 1, "cost_usd": 0.001, "requests": 1})
        return _fake_lane(ok=True)

    monkeypatch.setattr(bench_mod, "run_lane", successful)
    rc = main(["run", str(fixture_dir), "--runs", "2", "--out", str(out), "--allow-spend"])
    assert rc == 0
    assert not stale.exists()  # stale run files must not contaminate `score`
    assert not stale_progress.exists()
    assert not stale_progress_tmp.exists()
    assert not list(out.glob("progress-*"))
    assert sorted(p.name for p in out.glob("run-*.json")) == ["run-0.json", "run-1.json"]

    monkeypatch.setattr(bench_mod, "run_lane", lambda **kwargs: _fake_lane(ok=False))
    assert main(["run", str(fixture_dir), "--runs", "1", "--out", str(out), "--allow-spend"]) == 1

    assert main(["run", str(fixture_dir), "--runs", "0", "--out", str(out), "--allow-spend"]) == 1


def test_cmd_run_wires_provider_data_policy_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import bench as bench_mod

    fixture_dir = tmp_path / "f"
    _write_fixture(fixture_dir, [])
    out = tmp_path / "out"
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    calls: list[dict] = []

    def successful(**kwargs):
        calls.append(kwargs)
        return _fake_lane(ok=True)

    monkeypatch.setattr(bench_mod, "run_lane", successful)
    assert (
        main(
            [
                "run",
                str(fixture_dir),
                "--out",
                str(out),
                "--provider-data-collection",
                "deny",
                "--provider-zdr",
                "--allow-spend",
            ]
        )
        == 0
    )
    assert calls[0]["provider_data_collection"] == "deny"
    assert calls[0]["provider_zdr"] is True

    calls.clear()
    assert main(["run", str(fixture_dir), "--out", str(out), "--allow-spend"]) == 0
    assert calls[0]["provider_data_collection"] == "deny"
    assert calls[0]["provider_zdr"] is True

    calls.clear()
    assert (
        main(
            [
                "run",
                str(fixture_dir),
                "--out",
                str(out),
                "--provider-data-collection",
                "allow",
                "--no-provider-zdr",
                "--allow-spend",
            ]
        )
        == 0
    )
    assert calls[0]["provider_data_collection"] == "allow"
    assert calls[0]["provider_zdr"] is False


def test_cmd_run_wires_benchmark_lane_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import bench as bench_mod

    fixture_dir = tmp_path / "f"
    _write_fixture(fixture_dir, [])
    out = tmp_path / "out"
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    calls: list[dict] = []

    def successful(**kwargs):
        calls.append(kwargs)
        return _fake_lane(ok=True)

    monkeypatch.setattr(bench_mod, "run_lane", successful)

    assert (
        main(
            [
                "run",
                str(fixture_dir),
                "--out",
                str(out),
                "--lane-timeout",
                "1680",
                "--allow-spend",
            ]
        )
        == 0
    )
    assert calls[0]["lane_timeout"] == 1680

    calls.clear()
    monkeypatch.setenv("OR_PR_REVIEW_BENCH_LANE_TIMEOUT_SECONDS", "1740")
    assert main(["run", str(fixture_dir), "--out", str(out), "--allow-spend"]) == 0
    assert calls[0]["lane_timeout"] == 1740

    monkeypatch.setenv("OR_PR_REVIEW_BENCH_LANE_TIMEOUT_SECONDS", "invalid")
    assert main(["run", str(fixture_dir), "--out", str(out), "--allow-spend"]) == 1


def test_cmd_run_preserves_aggregate_progress_when_lane_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import bench as bench_mod

    fixture_dir = tmp_path / "f"
    _write_fixture(fixture_dir, [])
    out = tmp_path / "out"
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    def interrupted(**kwargs):
        kwargs["progress"](
            {
                "elapsed_ms": 123,
                "cost_usd": 0.004,
                "requests": 2,
                "provider": "example",
            }
        )
        raise ActionError("interrupted")

    monkeypatch.setattr(bench_mod, "run_lane", interrupted)
    assert main(["run", str(fixture_dir), "--out", str(out), "--allow-spend"]) == 1
    progress = json.loads((out / "progress-0.json").read_text(encoding="utf-8"))
    assert progress == {
        "schema": "or-pr-review/bench-progress/1",
        "model": "x-ai/grok-4.6",
        "elapsed_ms": 123,
        "cost_usd": 0.004,
        "requests": 2,
        "provider": "example",
    }


def test_cmd_score_reports_means_and_skips_failed_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture_dir = tmp_path / "f"
    _write_fixture(
        fixture_dir,
        [
            {
                "id": "B1",
                "severity": "bug",
                "file": "a.py",
                "context": "diff",
                "keywords": ["KeyError"],
            },
            {
                "id": "R9",
                "severity": "risk",
                "file": "docs/map.md",
                "context": "repo",
                "keywords": ["inventory"],
            },
        ],
    )
    (fixture_dir / "adjudications.json").write_text(
        json.dumps(
            [
                {
                    "id": "A1",
                    "verdict": "true_positive_unlabeled",
                    "file": None,
                    "keywords": ["missing docstring"],
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    out.mkdir()
    good = {
        "ok": True,
        "findings": [
            {"file": "a.py", "title": "KeyError", "body": "", "severity": "bug"},
            {
                "file": "docs/map.md",
                "title": "inventory row absent",
                "body": "",
                "severity": "risk",
            },
            {"file": "b.py", "title": "Missing docstring", "body": "", "severity": "nit"},
            {"file": "c.py", "title": "mystery", "body": "", "severity": "nit"},
        ],
    }
    (out / "run-0.json").write_text(json.dumps(good), encoding="utf-8")
    (out / "run-3.json").write_text(json.dumps({"ok": False, "error": "x"}), encoding="utf-8")
    assert main(["score", str(fixture_dir), str(out)]) == 0
    printed = capsys.readouterr().out
    assert "1 scored run(s), 1 failed run(s)" in printed
    assert "run-0 | 2/2" in printed  # rows named by file, not enumeration
    assert "mean | 100%" in printed
    # The new instrumentation must actually render — fixture.adjudications
    # threaded through _cmd_score, strata line, frequency line, tags.
    assert "recall by context: diff 1/1 (100%), repo 1/1 (100%)" in printed
    assert "label detection frequency: B1 1/1, R9 1/1" in printed
    assert "<adjudicated TP:A1>" in printed
    assert "<UNADJUDICATED" in printed
    # noise column counts adjudicated-FP + unadjudicated (here: 1 of 4).
    assert "| 3/3 | 1/4 | 4" in printed


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (b"\xff", "not valid UTF-8"),
        (b"{", "not valid JSON"),
        (b"[]", "must contain a JSON object"),
    ],
)
def test_cmd_score_reports_clean_json_read_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    contents: bytes,
    message: str,
) -> None:
    fixture_dir = tmp_path / "fixture"
    _write_fixture(fixture_dir, [])
    out = tmp_path / "out"
    out.mkdir()
    (out / "run-0.json").write_bytes(contents)

    assert main(["score", str(fixture_dir), str(out)]) == 1
    assert message in capsys.readouterr().err


def test_cmd_score_reports_json_read_os_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture_dir = tmp_path / "fixture"
    _write_fixture(fixture_dir, [])
    out = tmp_path / "out"
    out.mkdir()
    result = out / "run-0.json"
    result.write_text("{}", encoding="utf-8")
    original_read_bytes = Path.read_bytes

    def fail_result_read(path: Path) -> bytes:
        if path == result:
            raise PermissionError("fixture is not readable")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_result_read)
    assert main(["score", str(fixture_dir), str(out)]) == 1
    assert "could not read JSON file" in capsys.readouterr().err


def test_cmd_score_clean_fixture_reports_noise(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture_dir = tmp_path / "clean"
    _write_fixture(fixture_dir, [])
    out = tmp_path / "out"
    out.mkdir()
    payload = {
        "ok": True,
        "findings": [{"file": "a.py", "title": "phantom", "body": "", "severity": "nit"}],
    }
    (out / "run-0.json").write_text(json.dumps(payload), encoding="utf-8")
    assert main(["score", str(fixture_dir), str(out)]) == 0
    printed = capsys.readouterr().out
    assert "clean fixture (no labels): the noise column is the score" in printed
    assert "| 1/1 | 1" in printed  # noise 1/1, findings 1


def _expected_judge_output(fixture: JudgeFixture) -> list[dict]:
    return [
        {
            "title": expected.title,
            "body": expected.keywords[0],
            "severity": "risk",
            "file": expected.file,
            "line": expected.line,
            "models": ["synthetic/reviewer-a"],
        }
        for expected in fixture.expected_issues
    ]


def test_judge_fixture_hash_reuses_the_bytes_read_for_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_path = JUDGE_FIXTURE_DIR / "ab-duplicates-complements.json"
    expected_bytes = fixture_path.read_bytes()
    original_read_bytes = Path.read_bytes
    reads = 0

    def counting_read(path: Path) -> bytes:
        nonlocal reads
        if path == fixture_path:
            reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counting_read)
    fixture = load_judge_fixture(fixture_path)
    assert reads == 1
    assert fixture.sha256 == hashlib.sha256(expected_bytes).hexdigest()


def test_committed_judge_fixtures_cover_ab_abc_and_hygiene_cases() -> None:
    fixtures = {
        path.stem: load_judge_fixture(path) for path in sorted(JUDGE_FIXTURE_DIR.glob("*.json"))
    }
    assert set(fixtures) == {
        "ab-clean-diagnostics",
        "ab-duplicates-complements",
        "abc-conflicts-minority",
    }
    assert len(fixtures["ab-duplicates-complements"].lanes) == 2
    abc = fixtures["abc-conflicts-minority"]
    assert len(abc.lanes) == 3
    assert sum(issue.recall_critical for issue in abc.expected_issues) == 1
    exact_anchor = [
        finding
        for lane in abc.lanes
        for finding in lane["findings"]
        if finding["title"] == "Missing validation"
        and finding["file"] == "src/api.py"
        and finding["line"] == 42
    ]
    assert len(exact_anchor) == 2
    assert "negative age" in exact_anchor[0]["body"]
    assert "unauthorized account" in exact_anchor[1]["body"]
    clean = fixtures["ab-clean-diagnostics"]
    assert clean.expected_verdict == "clean"
    assert clean.expected_issues == ()


def test_judge_fixtures_exercise_production_hygiene_offline() -> None:
    from or_pr_review.bench import _merged_issue_dict
    from or_pr_review.judge import deterministic_union, partition_reviewable_lanes

    clean = load_judge_fixture(JUDGE_FIXTURE_DIR / "ab-clean-diagnostics.json")
    clean_lanes, clean_diagnostics = partition_reviewable_lanes(list(clean.lanes))
    assert len(clean_diagnostics) == 2
    assert deterministic_union(clean_lanes) == []
    assert score_judge_output([], clean).verdict_correct

    abc = load_judge_fixture(JUDGE_FIXTURE_DIR / "abc-conflicts-minority.json")
    abc_lanes, abc_diagnostics = partition_reviewable_lanes(list(abc.lanes))
    assert len(abc_diagnostics) == 1
    baseline = [_merged_issue_dict(issue) for issue in deterministic_union(abc_lanes)]
    score = score_judge_output(baseline, abc)
    assert score.recall == 1.0  # fallback preserves the minority issue
    assert score.critical_hit == score.critical_total == 1
    assert score.duplicate_count >= 1  # semantic merging is where the LLM adds value
    assert (
        sum(
            issue["title"] == "Missing validation"
            and issue["file"] == "src/api.py"
            and issue["line"] == 42
            for issue in baseline
        )
        == 2
    )


def test_judge_scorer_reports_recall_precision_duplicates_and_verdict() -> None:
    fixture = load_judge_fixture(JUDGE_FIXTURE_DIR / "ab-duplicates-complements.json")
    issues = _expected_judge_output(fixture)
    # A second rendering of the retry issue is a duplicate. The unrelated
    # tool limitation is unsupported/noise rather than another code issue.
    issues.extend(
        [
            {
                "title": "401 retry loop remains",
                "body": "A permanent 401 is requeued forever.",
                "severity": "bug",
                "file": "src/jobs.py",
                "line": 44,
                "models": ["synthetic/reviewer-b"],
            },
            {
                "title": "Registry could not be inspected",
                "body": "Review tooling failed.",
                "severity": "risk",
                "file": "config/registry.json",
                "line": None,
                "models": ["synthetic/reviewer-b"],
            },
        ]
    )
    score = score_judge_output(issues, fixture)
    assert score.recall == 1.0
    assert score.precision == 0.5
    assert score.duplicate_count == 1
    assert score.duplicate_rate == 0.25
    assert score.unmatched_titles == ("Registry could not be inspected",)
    assert score.verdict_correct


def test_judge_scorer_uses_one_to_one_matching_and_tracks_critical_recall() -> None:
    fixture = load_judge_fixture(JUDGE_FIXTURE_DIR / "abc-conflicts-minority.json")
    issues = _expected_judge_output(fixture)
    critical = next(issue for issue in fixture.expected_issues if issue.recall_critical)
    issues = [issue for issue in issues if issue["title"] != critical.title]
    score = score_judge_output(issues, fixture)
    assert score.recall == 7 / 8
    assert score.critical_hit == 0
    assert score.critical_total == 1
    assert score.missed_ids == ("AUTH-AFTER-CACHE",)

    # One broad row mentioning both retry symptoms can cover only one of the
    # two distinct expected defects at the same location.
    retries_only = [
        {
            "title": "Retry policy mishandles 404 and 503",
            "body": "Permanent 404 is retried while transient 503 is not retried.",
            "severity": "risk",
            "file": "src/client.py",
            "line": 72,
            "models": ["synthetic/reviewer-a"],
        }
    ]
    broad_score = score_judge_output(retries_only, fixture)
    assert broad_score.true_positive_count == 1

    # Exact title/file/line equality is not enough to merge: these are two
    # distinct behaviors and one combined output must lose recall.
    same_anchor_ids = {"NEGATIVE-AGE-API", "UNAUTHORIZED-ACCOUNT-API"}
    other_issues = [
        issue
        for issue, expected in zip(
            _expected_judge_output(fixture), fixture.expected_issues, strict=True
        )
        if expected.id not in same_anchor_ids
    ]
    overmerged = {
        "title": "Missing validation",
        "body": (
            "Negative age=-1 is accepted, and an account_id owned by another user "
            "is accepted without authorization."
        ),
        "severity": "bug",
        "file": "src/api.py",
        "line": 42,
        "models": ["synthetic/reviewer-a", "synthetic/reviewer-b"],
    }
    overmerge_score = score_judge_output([*other_issues, overmerged], fixture)
    assert overmerge_score.recall == 7 / 8
    assert len(overmerge_score.missed_ids) == 1
    assert overmerge_score.missed_ids[0] in same_anchor_ids


def test_judge_clean_fixture_scores_false_issue_and_verdict() -> None:
    fixture = load_judge_fixture(JUDGE_FIXTURE_DIR / "ab-clean-diagnostics.json")
    clean = score_judge_output([], fixture)
    assert clean.precision == clean.recall == 1.0
    assert clean.verdict_correct
    noisy = score_judge_output(
        [{"title": "tool failed", "body": "", "file": None, "line": None}], fixture
    )
    assert noisy.precision == 0.0
    assert noisy.recall == 1.0  # zero labels; noise is measured by precision/verdict
    assert not noisy.verdict_correct


def test_judge_run_is_double_gated_before_any_live_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import bench as bench_mod

    called = False

    def fake_judge(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("must not call")

    monkeypatch.setattr(bench_mod, "run_llm_judge", fake_judge)
    fixture = JUDGE_FIXTURE_DIR / "ab-duplicates-complements.json"
    out = tmp_path / "out"
    assert main(["judge-run", str(fixture), "--out", str(out)]) == 1
    assert not called
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert (
        main(
            [
                "judge-run",
                str(fixture),
                "--out",
                str(out),
                "--allow-spend",
            ]
        )
        == 1
    )
    assert not called


def test_judge_run_records_repeatability_metadata_without_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import bench as bench_mod
    from or_pr_review.merge import MergedIssue

    calls = []

    def fake_judge(**kwargs):
        calls.append(kwargs)
        return (
            [
                MergedIssue(
                    title="Permanent failures retry forever",
                    body="A permanent 401 requeues forever.",
                    severity="bug",
                    file="src/jobs.py",
                    line=42,
                    models=["synthetic/reviewer-a", "synthetic/reviewer-b"],
                )
            ],
            "merged",
            0.000123,
        )

    monkeypatch.setattr(bench_mod, "run_llm_judge", fake_judge)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret-must-not-be-recorded")
    fixture = JUDGE_FIXTURE_DIR / "ab-duplicates-complements.json"
    out = tmp_path / "out"
    assert (
        main(
            [
                "judge-run",
                str(fixture),
                "--models",
                "google/gemini-3.1-flash-lite,openai/gpt-5.6-luna",
                "--runs",
                "1",
                "--out",
                str(out),
                "--allow-spend",
            ]
        )
        == 0
    )
    assert [call["model"] for call in calls] == [
        "google/gemini-3.1-flash-lite",
        "openai/gpt-5.6-luna",
    ]
    assert all(call["provider_data_collection"] == "deny" for call in calls)
    assert all(call["provider_zdr"] is True for call in calls)
    results = sorted(out.glob("judge-*.json"))
    assert len(results) == 2
    for result in results:
        text = result.read_text(encoding="utf-8")
        assert "sk-secret" not in text
        payload = json.loads(text)
        assert payload["fixture_sha256"]
        assert payload["prompt_sha256"]
        assert payload["harness_version"]
        assert payload["production_default_judge"] == "openai/gpt-5.6-luna"
        assert payload["cost_usd"] == 0.000123


def test_judge_run_preflights_all_output_paths_before_first_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import bench as bench_mod

    calls = []
    monkeypatch.setattr(bench_mod, "run_llm_judge", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    fixture = JUDGE_FIXTURE_DIR / "ab-duplicates-complements.json"

    # The second model's output already exists. A sequential preflight would
    # spend on Gemini before noticing it; the complete matrix must abort first.
    out = tmp_path / "existing-later"
    out.mkdir()
    (out / "judge-ab-duplicates-complements-openai-gpt-5.6-luna-run-0.json").write_text(
        "{}", encoding="utf-8"
    )
    assert (
        main(
            [
                "judge-run",
                str(fixture),
                "--models",
                "google/gemini-3.1-flash-lite,openai/gpt-5.6-luna",
                "--runs",
                "1",
                "--out",
                str(out),
                "--allow-spend",
            ]
        )
        == 1
    )
    assert calls == []

    # Distinct valid slugs can sanitize to the same filename. Reject the
    # collision before creating the output directory or making a call.
    collision_out = tmp_path / "sanitized-collision"
    assert (
        main(
            [
                "judge-run",
                str(fixture),
                "--models",
                "vendor/a:b,vendor-a/b",
                "--runs",
                "1",
                "--out",
                str(collision_out),
                "--allow-spend",
            ]
        )
        == 1
    )
    assert not collision_out.exists()
    assert calls == []


def test_judge_run_caps_runs_before_directory_creation_or_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import bench as bench_mod

    calls = []
    monkeypatch.setattr(bench_mod, "run_llm_judge", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    out = tmp_path / "too-many-runs"
    assert (
        main(
            [
                "judge-run",
                str(JUDGE_FIXTURE_DIR / "ab-duplicates-complements.json"),
                "--runs",
                "21",
                "--out",
                str(out),
                "--allow-spend",
            ]
        )
        == 1
    )
    assert calls == []
    assert not out.exists()


def test_judge_run_redacts_failed_call_in_console_and_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from or_pr_review import bench as bench_mod

    secret = "sk-or-v1-do-not-leak"

    def fail_judge(**kwargs):
        raise ActionError(f"upstream rejected api_key={secret}")

    monkeypatch.setattr(bench_mod, "run_llm_judge", fail_judge)
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    out = tmp_path / "failed"
    assert (
        main(
            [
                "judge-run",
                str(JUDGE_FIXTURE_DIR / "ab-duplicates-complements.json"),
                "--runs",
                "1",
                "--out",
                str(out),
                "--allow-spend",
            ]
        )
        == 1
    )
    printed = capsys.readouterr()
    assert secret not in printed.out
    assert secret not in printed.err
    result = next(out.glob("judge-*.json"))
    text = result.read_text(encoding="utf-8")
    assert secret not in text
    assert "[redacted]" in text


def test_judge_score_command_is_offline_and_rejects_fixture_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture_path = JUDGE_FIXTURE_DIR / "abc-conflicts-minority.json"
    fixture = load_judge_fixture(fixture_path)
    result = tmp_path / "judge-saved.json"
    result.write_text(
        json.dumps(
            {
                "ok": True,
                "fixture_sha256": fixture.sha256,
                "model": "saved/test-model",
                "mode": "merged",
                "elapsed_ms": 12,
                "issues": _expected_judge_output(fixture),
            }
        ),
        encoding="utf-8",
    )
    assert main(["judge-score", str(fixture_path), str(result)]) == 0
    printed = capsys.readouterr().out
    assert "8/8 | 8/8 | 0/8 | 1/1 | yes (issues)" in printed
    assert "mean recall: 100.0%" in printed

    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["fixture_sha256"] = "0" * 64
    result.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["judge-score", str(fixture_path), str(result)]) == 1
