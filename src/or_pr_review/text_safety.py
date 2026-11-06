"""Keep external text separate from harness markers and runner log commands."""

from __future__ import annotations

import json
import re
from typing import TextIO

from or_pr_review.redaction import redact

_RESERVED_MARKER_RE = re.compile(r"<!--\s*(?:openrouter-|or-finding:)", re.IGNORECASE)


def neutralize_markers(text: str) -> str:
    """Render reserved comment prefixes as text, leaving ordinary Markdown intact."""
    return _RESERVED_MARKER_RE.sub(lambda match: match[0].replace("<", "&lt;", 1), text)


def log_field(text: str) -> str:
    """Encode a redacted field on one line without runner command delimiters."""
    encoded = json.dumps(redact(text), ensure_ascii=True)[1:-1]
    return re.sub(r":{2,}", lambda match: " ".join(match[0]), encoded).replace("##[", "## [")


def log(*values: object, file: TextIO | None = None, flush: bool = False) -> None:
    """Print diagnostics through the same boundary on stdout and stderr."""
    print(log_field(" ".join(str(value) for value in values)), file=file, flush=flush)
