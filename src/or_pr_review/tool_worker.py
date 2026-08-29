"""Killable subprocess entry point for one inert repository tool call."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from or_pr_review.workspace import dispatch_tool


def main() -> int:
    if len(sys.argv) != 3:
        return 2
    try:
        arguments = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return 2
    if not isinstance(arguments, dict):
        return 2
    result = dispatch_tool(Path(sys.argv[1]), sys.argv[2], arguments)
    sys.stdout.write(json.dumps({"result": result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
