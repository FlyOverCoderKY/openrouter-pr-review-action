"""Local policy preview without credentials, network, or paid model calls."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from or_pr_review.errors import ActionError
from or_pr_review.review_policy import resolve_policy


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="or_pr_review policy")
    commands = parser.add_subparsers(dest="command", required=True)
    explain = commands.add_parser("explain", help="Preview effective target-branch policy")
    explain.add_argument("--repo", type=Path, default=Path.cwd())
    explain.add_argument("--base", required=True, help="Full immutable target commit SHA")
    explain.add_argument("--head", required=True, help="Full immutable PR commit SHA")
    explain.add_argument("--carried-path", action="append", default=[])
    lint = commands.add_parser("lint", help="Validate proposed REVIEW.md files locally")
    lint.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    if args.command == "lint":
        from or_pr_review.review_policy import parse_policy_file

        for path in args.files:
            if path.name != "REVIEW.md":
                raise ActionError(f"{path}: expected canonical filename REVIEW.md")
            try:
                with path.open("rb") as stream:
                    raw = stream.read(16 * 1024 + 1)
            except OSError as exc:
                raise ActionError(f"cannot read {path}: {exc}") from exc
            # Local lint validates syntax only; tree discovery validates hierarchy.
            parse_policy_file("REVIEW.md", raw)
            print(f"{path}: syntax valid; use policy explain to check ancestry and scope")
        return 0
    policy = resolve_policy(args.repo, args.base, args.head, tuple(args.carried_path))
    data = asdict(policy)
    for item in data["files"]:
        item.pop("content")
    data["execution"] = "preview only: no model calls or profile execution"
    print(json.dumps(data, indent=2, ensure_ascii=True))
    return 0
