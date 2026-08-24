"""Never print API keys, tokens, or .env values."""

from __future__ import annotations

import os
import re

_SECRET_ENV_NAMES = (
    "OPENROUTER_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "XAI_API_KEY",
)

_KEY_ASSIGN_RE = re.compile(
    r"(?i)\b(OPENROUTER_API_KEY|GITHUB_TOKEN|GH_TOKEN|XAI_API_KEY|"
    r"api[_-]?key|token|secret|password|authorization)\b"
    r"(\s*[:=]\s*)(\S+)"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+\S+")
_ENV_LINE_RE = re.compile(
    r"(?m)^([A-Za-z_][A-Za-z0-9_]*)=([^\n]+)$",
)


def secret_values() -> list[str]:
    values: list[str] = []
    for name in _SECRET_ENV_NAMES:
        raw = os.environ.get(name, "")
        if raw:
            values.append(raw)
    return values


def redact(text: str, extra: list[str] | None = None) -> str:
    if not text:
        return text
    out = text
    for value in list(extra or []) + secret_values():
        if value and value in out:
            out = out.replace(value, "[redacted]")
    out = _BEARER_RE.sub("Bearer [redacted]", out)
    out = _KEY_ASSIGN_RE.sub(r"\1\2[redacted]", out)
    return out


def looks_like_dotenv(text: str) -> bool:
    """Heuristic: a block of KEY=value lines that should not be logged."""
    lines = [line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if len(lines) < 2:
        return False
    hits = sum(1 for line in lines if _ENV_LINE_RE.match(line))
    return hits >= 2 and hits / len(lines) >= 0.7
