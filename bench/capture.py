"""Capture a real PR as an offline bench fixture.

Fetches the PR's metadata and full diff with `gh`, verifies the PR head did
not advance between the two requests (so the diff, SHA, and checkout all
describe one revision), materializes that exact commit from a local clone
with `git archive`, and writes a fixture directory with an empty labels.json
for you to curate (golden labels = the adjudicated union of every reviewer's
validated findings).

Fixtures captured from private repositories contain private code: write them
to bench/fixtures-local/ (gitignored), never bench/fixtures/. Paths under
bench/fixtures/ are refused unless --allow-committed is passed.

Usage:
    python bench/capture.py --repo RetireGolden/RetireGolden --pr 331 \
        --clone ~/src/RetireGolden --out bench/fixtures-local/rg-331
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

TIMEOUT = 600


def run(*argv: str, cwd: Path | None = None) -> str:
    import os

    env = dict(os.environ)
    env.setdefault("GH_PROMPT_DISABLED", "1")
    env.setdefault("GH_NO_UPDATE_NOTIFIER", "1")
    try:
        result = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT, env=env
        )
    except subprocess.TimeoutExpired:
        raise SystemExit(f"command timed out after {TIMEOUT}s: {' '.join(argv)}")
    if result.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(argv)}\n{result.stderr.strip()}")
    return result.stdout


def pr_meta(repo: str, pr: int) -> dict:
    raw = run(
        "gh", "pr", "view", str(pr), "--repo", repo,
        "--json", "title,body,headRefOid,baseRefName,headRefName",
    )
    return json.loads(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--clone", required=True, help="local clone of the repo")
    parser.add_argument("--out", required=True, help="fixture directory to create")
    parser.add_argument(
        "--max-diff-kb",
        type=int,
        default=600,
        help="recorded in fixture.json; the bench applies the same embed cap as the workflow",
    )
    parser.add_argument(
        "--allow-committed",
        action="store_true",
        help="permit an --out under bench/fixtures/ (committable; private code must not go there)",
    )
    args = parser.parse_args()

    out = Path(args.out)
    resolved_out = out.resolve()
    committed_fixtures = (Path(__file__).resolve().parent / "fixtures").resolve()
    if not args.allow_committed and (
        resolved_out == committed_fixtures or committed_fixtures in resolved_out.parents
    ):
        raise SystemExit(
            f"{out} is inside the committed bench/fixtures/ tree; captured PRs can "
            "contain private code — use bench/fixtures-local/ (gitignored), or pass "
            "--allow-committed for a fixture that is genuinely safe to publish"
        )
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already exists and is not empty")
    clone = Path(args.clone).expanduser()
    if not (clone / ".git").exists():
        raise SystemExit(f"{clone} is not a git clone")

    meta = pr_meta(args.repo, args.pr)
    head_sha = meta["headRefOid"]

    diff = run(
        "gh", "api", f"repos/{args.repo}/pulls/{args.pr}",
        "-H", "Accept: application/vnd.github.diff",
    )

    # The diff endpoint always describes the CURRENT head. If the PR advanced
    # between the metadata request and the diff request, the SHA and diff
    # describe different revisions — re-check and ask for a retry.
    after = pr_meta(args.repo, args.pr)
    if after["headRefOid"] != head_sha:
        raise SystemExit(
            f"PR head advanced during capture ({head_sha[:12]} -> "
            f"{after['headRefOid'][:12]}); re-run to capture the new head"
        )

    # Fetch the pinned SHA itself, not the branch tip, so a concurrent push
    # cannot change what gets archived.
    run("git", "fetch", "-q", "origin", head_sha, cwd=clone)

    # Stage into a temp directory and move into place only on success, so a
    # failed capture never leaves a partial fixture blocking the retry.
    with tempfile.TemporaryDirectory(prefix="or-bench-capture.") as tmp:
        stage = Path(tmp) / "fixture"
        checkout = stage / "checkout"
        checkout.mkdir(parents=True)
        tar_path = Path(tmp) / "head.tar"
        run("git", "archive", "-o", str(tar_path), head_sha, cwd=clone)
        with tarfile.open(tar_path) as tar:
            tar.extractall(checkout, filter="data")
        (stage / "diff.patch").write_text(diff, encoding="utf-8", newline="\n")
        (stage / "fixture.json").write_text(
            json.dumps(
                {
                    "title": meta["title"],
                    "body": meta.get("body") or "",
                    "pr_number": args.pr,
                    "head_sha": head_sha,
                    "base_ref": meta["baseRefName"],
                    "head_ref": meta["headRefName"],
                    "custom_instructions": "",
                    "max_diff_kb": args.max_diff_kb,
                    "diff_file": "diff.patch",
                    "checkout_dir": "checkout",
                    "labels_file": "labels.json",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (stage / "labels.json").write_text("[]\n", encoding="utf-8")
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.rmdir()  # empty by the earlier check
        shutil.move(str(stage), str(out))

    print(f"captured {args.repo}#{args.pr} at {head_sha[:12]} into {out}")
    print("next: curate labels.json from the adjudicated review findings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
