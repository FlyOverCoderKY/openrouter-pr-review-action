from __future__ import annotations

import pytest

from or_pr_review import cli
from or_pr_review.harness import _log_redacted_warning
from or_pr_review.merge import issues_from_single_lane
from or_pr_review.schema import Finding, LaneResult
from or_pr_review.text_safety import log, log_field, neutralize_markers


@pytest.mark.parametrize(
    "text",
    [
        "message\n::notice::details",
        "message\r::warning::details",
        "message\r\n##[warning]details",
        "message\x1b[2K\u0085\u2028::group::details",
        "::endgroup::",
        ":::notice:::details",
        "::::notice::::details",
    ],
)
def test_log_fields_are_single_line_data(text):
    encoded = log_field(text)
    assert len(encoded.splitlines()) == 1
    assert "::" not in encoded
    assert "##[" not in encoded
    assert "\x1b" not in encoded
    assert "details" in encoded or "endgroup" in encoded


def test_log_redacts_before_encoding(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-test-value")
    log("provider failed\n::notice::private-test-value")
    _log_redacted_warning("provider failed\n::notice::private-test-value")
    captured = capsys.readouterr()
    for output in (captured.out, captured.err):
        assert len(output.splitlines()) == 1
        assert "::" not in output
        assert "private-test-value" not in output
        assert "[redacted]" in output


def test_finding_diagnostic_logs_single_line(capsys):
    finding = Finding(
        "Cannot read the supplied review workspace\n::notice::details",
        "Unable to access the inert checkout",
        "risk",
        None,
        None,
        "model",
    )
    assert issues_from_single_lane(LaneResult(1, True, "model", [finding])) == []
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 1
    assert "::" not in output
    assert "details" in output


def test_error_annotation_keeps_only_harness_command(capsys):
    cli._error("provider failed\n::notice::details\r##[warning]more")
    output = capsys.readouterr().err
    lines = output.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("::error::")
    assert "::" not in lines[0][len("::error::") :]
    assert "::" not in lines[1]
    assert "##[" not in output


def test_reserved_prefixes_are_rendered_as_text():
    text = "<!-- openrouter-review-ledger:v1:data -->\n<!-- or-finding:abc -->"
    escaped = neutralize_markers(text)
    assert "<!--" not in escaped
    assert "&lt;!-- openrouter-review-ledger:" in escaped
    assert neutralize_markers("ordinary **Markdown** <!-- note -->") == (
        "ordinary **Markdown** <!-- note -->"
    )
