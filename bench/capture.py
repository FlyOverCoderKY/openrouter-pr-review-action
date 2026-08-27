"""Capture a real PR as an offline bench fixture.

Fetches the PR's full diff and metadata with `gh`, materializes the reviewed
head commit from a local clone with `git archive`, and writes a fixture
directory with an empty labels.json for you to curate (golden labels =
the adjudicated union of every reviewer's validated findings).

Fixtures captured from private repositories contain private code: write them
to bench/fixtures-local/ (gitignored), never bench/fixtures/.

Usage:
    python bench/capture.py --repo RetireGolden/RetireGolden --pr 331 \
        --clone ~/src/RetireGolden --out bench/fixtures-local/rg-331
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def run(*argv: str, cwd: Path | None = None) -> str:
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(argv)}\n{result.stderr.strip()}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--clone", required=True, help="local clone of the repo")
    parser.add_argument("--out", required=True, help="fixture directory to create")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already exists and is not empty")
    clone = Path(args.clone).expanduser()
    if not (clone / ".git").exists():
        raise SystemExit(f"{clone} is not a git clone")

    meta_raw = run(
        "gh", "pr", "view", str(args.pr), "--repo", args.repo,
        "--json", "title,body,headRefOid,baseRefName,headRefName",
    )
    meta = json.loads(meta_raw)
    head_sha = meta["headRefOid"]

    diff = run(
        "gh", "api", f"repos/{args.repo}/pulls/{args.pr}",
        "-H", "Accept: application/vnd.github.diff",
    )

    run("git", "fetch", "-q", "origin", f"pull/{args.pr}/head", cwd=clone)
    checkout = out / "checkout"
    checkout.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "head.tar"
        run("git", "archive", "-o", str(tar_path), head_sha, cwd=clone)
        with tarfile.open(tar_path) as tar:
            tar.extractall(checkout, filter="data")

    (out / "diff.patch").write_text(diff, encoding="utf-8", newline="\n")
    (out / "fixture.json").write_text(
        json.dumps(
            {
                "title": meta["title"],
                "body": meta.get("body") or "",
                "pr_number": args.pr,
                "head_sha": head_sha,
                "base_ref": meta["baseRefName"],
                "head_ref": meta["headRefName"],
                "custom_instructions": "",
                "diff_file": "diff.patch",
                "checkout_dir": "checkout",
                "labels_file": "labels.json",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    labels = out / "labels.json"
    if not labels.exists():
        labels.write_text("[]\n", encoding="utf-8")
    print(f"captured {args.repo}#{args.pr} at {head_sha[:12]} into {out}")
    print("next: curate labels.json from the adjudicated review findings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
