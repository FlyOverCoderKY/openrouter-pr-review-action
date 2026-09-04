"""Killable subprocess entry point for one inert repository tool call.

The harness launches this module with ``sys.executable -m`` so it uses the
same Python environment in which ``or_pr_review`` is installed.  Keeping that
runtime dependency explicit matters for action runners that have multiple
Python installations: invoking a different interpreter can make the worker
fail before it can return a model-facing tool observation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from or_pr_review.workspace import dispatch_tool


def _diagnostic(message: str) -> None:
    """Write a concise operator diagnostic without contaminating JSON stdout."""
    print(f"tool_worker: {message}", file=sys.stderr, flush=True)


def main() -> int:
    if len(sys.argv) != 3:
        _diagnostic("usage: python -m or_pr_review.tool_worker WORKSPACE TOOL")
        return 2
    try:
        arguments = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        _diagnostic(f"invalid JSON arguments at character {exc.pos}")
        return 2
    if not isinstance(arguments, dict):
        _diagnostic("JSON arguments must be an object")
        return 2
    result = dispatch_tool(Path(sys.argv[1]), sys.argv[2], arguments)
    sys.stdout.write(json.dumps({"result": result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
