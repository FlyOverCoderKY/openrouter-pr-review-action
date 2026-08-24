from __future__ import annotations

from or_pr_review.redaction import looks_like_dotenv, redact


def test_redact_literal_secrets() -> None:
    text = redact("token=abc Authorization: Bearer super-secret", extra=["super-secret"])
    assert "super-secret" not in text
    assert "[redacted]" in text


def test_redact_env_values(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-please-hide")
    text = redact("calling with sk-or-v1-please-hide")
    assert "sk-or-v1-please-hide" not in text


def test_looks_like_dotenv() -> None:
    assert looks_like_dotenv("FOO=1\nBAR=2\nBAZ=3\n")
    assert not looks_like_dotenv("just a paragraph of review text\nwith two lines")
