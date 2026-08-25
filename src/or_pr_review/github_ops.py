"""GitHub CLI helpers for PR metadata, diffs, comments, and reviews."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from or_pr_review.errors import ActionError
from or_pr_review.redaction import redact

STATUS_MARKER = "<!-- openrouter-pr-review-status -->"


class GitHub:
    def __init__(
        self,
        *,
        token: str,
        repository: str,
        timeout: int = 120,
        runner: Any = None,
    ) -> None:
        if not token.strip():
            raise ActionError("github_token is empty")
        if not repository or "/" not in repository:
            raise ActionError("GITHUB_REPOSITORY must be owner/repo")
        if timeout < 1 or timeout > 600:
            raise ActionError("github_timeout_seconds must be an integer from 1 through 600")
        self.token = token
        self.repository = repository
        self.timeout = timeout
        self._runner = runner or _run_gh
        self.owner, self.repo = repository.split("/", 1)

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["GH_TOKEN"] = self.token
        env["GITHUB_TOKEN"] = self.token
        env["GH_PROMPT_DISABLED"] = "1"
        return env

    def _gh(self, *args: str, stdin: str | None = None) -> str:
        try:
            return self._runner(
                ["gh", *args],
                env=self._env(),
                timeout=self.timeout,
                stdin=stdin,
            )
        except ActionError as exc:
            raise ActionError(redact(str(exc))) from exc

    def pr_view(self, number: int) -> dict[str, object]:
        raw = self._gh(
            "pr",
            "view",
            str(number),
            "--repo",
            self.repository,
            "--json",
            "number,title,body,headRefOid,headRefName,baseRefName,url",
        )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ActionError(f"gh pr view returned non-JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ActionError("gh pr view returned a non-object")
        return parsed

    def pr_diff(self, number: int) -> str:
        return self._gh("pr", "diff", str(number), "--repo", self.repository)

    def compare_diff(self, before: str, after: str) -> str:
        """Raw unified diff for a verified linear fast-forward range.

        The JSON compare payload caps its files array at 300 entries with no
        truncation marker, so the diff itself is fetched with the raw diff
        media type instead. A non-fast-forward range (force-push, rebase)
        raises so the caller falls back to the single latest commit with a
        visible notice rather than silently reviewing a merge-base diff.
        """
        endpoint = f"repos/{self.repository}/compare/{before}...{after}"
        raw = self._gh("api", endpoint)
        comparison = _json_object(raw, f"compare {before[:12]}...{after[:12]}")
        status = comparison.get("status")
        behind_by = comparison.get("behind_by")
        if status != "ahead" or behind_by not in {0, None}:
            raise ActionError("commit comparison is not a linear fast-forward range")
        return self._gh("api", "-H", "Accept: application/vnd.github.diff", endpoint)

    def commit_diff(self, sha: str) -> str:
        """Raw unified diff for one commit (complete; no JSON files-array caps)."""
        return self._gh(
            "api",
            "-H",
            "Accept: application/vnd.github.diff",
            f"repos/{self.repository}/commits/{sha}",
        )

    def list_issue_comments(self, number: int) -> list[dict[str, Any]]:
        raw = self._gh(
            "api",
            f"repos/{self.repository}/issues/{number}/comments?per_page=100",
        )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ActionError(f"issue comments JSON failed: {exc}") from exc
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, dict)]

    def create_issue_comment(self, number: int, body: str) -> dict[str, Any]:
        raw = self._gh(
            "api",
            f"repos/{self.repository}/issues/{number}/comments",
            "--input",
            "-",
            stdin=json.dumps({"body": body}),
        )
        return _json_object(raw, "create comment")

    def update_issue_comment(self, comment_id: int, body: str) -> dict[str, Any]:
        raw = self._gh(
            "api",
            "-X",
            "PATCH",
            f"repos/{self.repository}/issues/comments/{comment_id}",
            "--input",
            "-",
            stdin=json.dumps({"body": body}),
        )
        return _json_object(raw, "update comment")

    def create_review(self, number: int, body: str, commit_id: str) -> dict[str, Any]:
        raw = self._gh(
            "api",
            f"repos/{self.repository}/pulls/{number}/reviews",
            "--input",
            "-",
            stdin=json.dumps({"event": "COMMENT", "body": body, "commit_id": commit_id}),
        )
        return _json_object(raw, "create review")


def _json_object(raw: str, what: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ActionError(f"{what} returned non-JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ActionError(f"{what} returned a non-object")
    return parsed


def _run_gh(
    cmd: list[str],
    *,
    env: dict[str, str],
    timeout: int,
    stdin: str | None = None,
) -> str:
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            input=stdin.encode("utf-8") if stdin is not None else None,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ActionError(f"GitHub CLI timed out after {timeout}s") from exc
    except OSError as exc:
        raise ActionError(f"failed to run GitHub CLI: {exc}") from exc
    if proc.returncode != 0:
        err = redact((proc.stderr or b"").decode("utf-8", errors="replace")[:600])
        raise ActionError(f"GitHub CLI failed ({' '.join(cmd[1:4])}): {err}")
    return proc.stdout.decode("utf-8")


def upsert_status_comment(
    github: GitHub,
    *,
    pr_number: int,
    body: str,
    enabled: bool,
) -> str | None:
    if not enabled:
        return None
    text = f"{STATUS_MARKER}\n{body}".rstrip() + "\n"
    for comment in github.list_issue_comments(pr_number):
        existing = comment.get("body")
        if isinstance(existing, str) and STATUS_MARKER in existing:
            comment_id = comment.get("id")
            if isinstance(comment_id, int):
                updated = github.update_issue_comment(comment_id, text)
                url = updated.get("html_url")
                return url if isinstance(url, str) else None
    created = github.create_issue_comment(pr_number, text)
    url = created.get("html_url")
    return url if isinstance(url, str) else None
