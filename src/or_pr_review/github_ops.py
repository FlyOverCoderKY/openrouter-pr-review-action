"""GitHub CLI helpers for PR metadata, diffs, comments, and reviews."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tempfile
from typing import Any

from or_pr_review.errors import ActionError, DivergedRangeError, InputLimitError
from or_pr_review.loop import FINDING_MARKER_RE
from or_pr_review.redaction import redact
from or_pr_review.text_safety import log

STATUS_MARKER = "<!-- openrouter-pr-review-status -->"
# The harness's own posted bodies (continuation parts, incomplete notices)
# must never be fed back to a verify round as "fixing agent responses".
_BOT_BODY_PREFIXES = (
    "## OpenRouter pull-request review",
    "## OpenRouter review incomplete",
)
_FULL_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
# Collection ceilings are independent of the caller's prompt embed budget.
MAX_COMMAND_BYTES = 16 * 1024 * 1024
MAX_DIFF_BYTES = 16 * 1024 * 1024
MAX_DIFF_FILES = 3_000
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
MAX_LIST_ITEMS = 10_000


class GitHub:
    def __init__(
        self,
        *,
        token: str,
        repository: str,
        timeout: int = 120,
        runner: Any = None,
        git_runner: Any = None,
        source_workspace: str | None = None,
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
        self._git_runner = git_runner or _run_git
        self.source_workspace = source_workspace
        self.owner, self.repo = repository.split("/", 1)

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        # The GitHub subprocess has no reason to receive the paid-provider
        # credential. Keep the rest of the runner environment intact while
        # narrowing this command's secret exposure to its GitHub token.
        env.pop("OPENROUTER_API_KEY", None)
        env["GH_TOKEN"] = self.token
        env["GITHUB_TOKEN"] = self.token
        env["GH_PROMPT_DISABLED"] = "1"
        return env

    def _gh(self, *args: str, stdin: str | None = None) -> str:
        if stdin is not None:
            _check_bytes(stdin, MAX_PAYLOAD_BYTES, "GitHub request")
        try:
            raw = self._runner(
                ["gh", *args],
                env=self._env(),
                timeout=self.timeout,
                stdin=stdin,
            )
        except InputLimitError:
            raise
        except ActionError as exc:
            raise ActionError(redact(str(exc))) from exc
        _check_bytes(raw, MAX_COMMAND_BYTES, "GitHub response")
        return raw

    def pr_view(self, number: int) -> dict[str, object]:
        raw = self._gh(
            "pr",
            "view",
            str(number),
            "--repo",
            self.repository,
            "--json",
            "number,title,body,headRefOid,headRefName,baseRefOid,baseRefName,url",
        )
        return _json_object(raw, "gh pr view")

    def pr_diff(
        self,
        number: int,
        *,
        base_sha: str | None = None,
        head_sha: str | None = None,
    ) -> str:
        # Both the API request and local fallback name immutable objects.
        # Metadata confirmation in collect remains useful for stale reviews.
        base = require_full_sha(base_sha, "base")
        head = require_full_sha(head_sha, "head")
        endpoint = f"repos/{self.repository}/compare/{base}...{head}"
        try:
            raw = self._gh("api", "-H", "Accept: application/vnd.github.diff", endpoint)
            return _checked_diff(raw)
        except InputLimitError:
            raise
        except ActionError:
            if not self.source_workspace:
                raise

        # Keep the same pinned range for any API collection failure. A full
        # checkout can serve fork objects and ranges unavailable through the API.
        workspace = self.source_workspace
        log(
            "notice: GitHub could not serve the pinned PR diff; "
            "falling back to local git diff base...head"
        )
        try:
            raw = self._git_runner(
                [
                    "git",
                    "diff",
                    "--no-color",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--find-renames",
                    f"{base}...{head}",
                    "--",
                ],
                timeout=self.timeout,
                cwd=workspace,
            )
        except InputLimitError:
            raise
        except ActionError as exc:
            raise ActionError(redact(str(exc))) from exc
        return _checked_diff(raw)

    def compare_diff(self, before: str, after: str) -> str:
        """Raw unified diff for a verified linear fast-forward range.

        The JSON compare payload caps its files array at 300 entries with no
        truncation marker, so the diff itself is fetched with the raw diff
        media type instead. A non-fast-forward range (force-push, rebase)
        raises so the caller falls back to the single latest commit with a
        visible notice rather than silently reviewing a merge-base diff.
        """
        endpoint = f"repos/{self.repository}/compare/{before}...{after}"
        # GitHub includes file patches only on page 1. Request one commit
        # from page 2: range status is unchanged, without duplicating diff data.
        raw = self._gh("api", f"{endpoint}?per_page=1&page=2")
        comparison = _json_object(raw, f"compare {before[:12]}...{after[:12]}")
        status = comparison.get("status")
        behind_by = comparison.get("behind_by")
        if status != "ahead" or behind_by not in {0, None}:
            raise DivergedRangeError("commit comparison is not a linear fast-forward range")
        return _checked_diff(self._gh("api", "-H", "Accept: application/vnd.github.diff", endpoint))

    def commit_diff(self, sha: str) -> str:
        """Raw unified diff for one commit (complete; no JSON files-array caps)."""
        return _checked_diff(
            self._gh(
                "api",
                "-H",
                "Accept: application/vnd.github.diff",
                f"repos/{self.repository}/commits/{sha}",
            )
        )

    def _paginated_list(self, endpoint: str, label: str) -> list[dict[str, Any]]:
        raw = self._gh("api", "--paginate", "--slurp", endpoint)
        _check_bytes(raw, MAX_PAYLOAD_BYTES, label)
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ActionError(f"could not parse {label}") from exc
        if not isinstance(payload, list):
            return []
        pages = (
            payload if payload and all(isinstance(page, list) for page in payload) else [payload]
        )
        if len(pages) > MAX_LIST_ITEMS or sum(len(page) for page in pages) > MAX_LIST_ITEMS:
            raise InputLimitError(f"{label} exceeds the {MAX_LIST_ITEMS} item collection limit")
        items = [item for page in pages for item in page]
        return [item for item in items if isinstance(item, dict)]

    def list_bot_review_bodies(self, number: int, bot_login: str) -> list[str]:
        """Review bodies authored by the action's own identity, oldest first.

        Only these bodies may carry loop-ledger state: any PR reviewer can
        post a review, so bodies from other authors are untrusted and never
        scanned.
        """
        reviews = self._paginated_list(f"repos/{self.repository}/pulls/{number}/reviews", "reviews")
        return [
            body
            for review in reviews
            if _comment_login(review) == bot_login and isinstance(body := review.get("body"), str)
        ]

    def list_finding_replies(
        self, number: int, *, generation: str, bot_login: str, carried_ids: set[str]
    ) -> list[tuple[str, str, str]]:
        """(finding_id, login, body) for replies to the bot's inline comments.

        Only markers from the given loop generation pair: finding ids restart
        at r1-1 after a loop reset, so an old generation's threads must never
        be attributed to a new finding that reuses the id.
        """
        if not generation:
            return []
        comments = self._paginated_list(
            f"repos/{self.repository}/pulls/{number}/comments", "review comments"
        )
        finding_by_comment_id: dict[int, str] = {}
        for comment in comments:
            ident = comment.get("id")
            body = comment.get("body")
            if _comment_login(comment) != bot_login or comment.get("in_reply_to_id") is not None:
                continue
            if isinstance(ident, int) and isinstance(body, str):
                match = FINDING_MARKER_RE.match(body)
                if match and match.group(1) == generation and match.group(2) in carried_ids:
                    finding_by_comment_id[ident] = match.group(2)
        replies: list[tuple[str, str, str]] = []
        for comment in comments:
            parent = comment.get("in_reply_to_id")
            body = comment.get("body")
            if not isinstance(parent, int) or not isinstance(body, str):
                continue
            finding_id = finding_by_comment_id.get(parent)
            if finding_id is None or FINDING_MARKER_RE.search(body):
                continue
            replies.append((finding_id, _comment_login(comment), body))
        return replies

    def list_rebase_replies(self, number: int, bot_login: str) -> list[tuple[str, str, str]]:
        """Replies across generations, anchored to actual bot-authored findings."""
        comments = self._paginated_list(
            f"repos/{self.repository}/pulls/{number}/comments", "review comments"
        )
        roots: dict[int, str] = {}
        for comment in comments:
            body, ident = comment.get("body"), comment.get("id")
            if (
                _comment_login(comment) != bot_login
                or not isinstance(ident, int)
                or comment.get("in_reply_to_id") is not None
            ):
                continue
            match = FINDING_MARKER_RE.match(body) if isinstance(body, str) else None
            if match:
                roots[ident] = f"{match.group(1)}/{match.group(2)}"
        return [
            (roots[comment["in_reply_to_id"]], _comment_login(comment), comment["body"])
            for comment in comments
            if isinstance(comment.get("in_reply_to_id"), int)
            and comment["in_reply_to_id"] in roots
            and isinstance(comment.get("body"), str)
        ]

    def list_recent_issue_comments(self, number: int, limit: int = 30) -> list[tuple[str, str]]:
        """(login, body) for the newest PR conversation comments, oldest first."""
        comments = self._paginated_list(
            f"repos/{self.repository}/issues/{number}/comments", "issue comments"
        )
        recent: list[tuple[str, str]] = []
        for comment in comments:
            body = comment.get("body")
            if not isinstance(body, str) or STATUS_MARKER in body:
                continue
            if body.startswith(_BOT_BODY_PREFIXES):
                continue
            recent.append((_comment_login(comment), body))
        return recent[-limit:]

    def list_issue_comments(self, number: int) -> list[dict[str, Any]]:
        return self._paginated_list(
            f"repos/{self.repository}/issues/{number}/comments", "issue comments"
        )

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

    def create_review(
        self,
        number: int,
        body: str,
        commit_id: str,
        comments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"event": "COMMENT", "body": body, "commit_id": commit_id}
        if comments:
            payload["comments"] = comments
        try:
            raw = self._gh(
                "api",
                f"repos/{self.repository}/pulls/{number}/reviews",
                "--input",
                "-",
                stdin=json.dumps(payload),
            )
        except ActionError as exc:
            if not comments or not _is_inline_comment_validation_error(str(exc)):
                raise
            # Inline placement can fail (for example a line outside the diff
            # hunks); the review body itself must still post — but never
            # silently.
            log(
                f"warning: inline comments were rejected and dropped "
                f"({len(comments)} comment(s)): {redact(str(exc))}"
            )
            fallback = {"event": "COMMENT", "body": body, "commit_id": commit_id}
            raw = self._gh(
                "api",
                f"repos/{self.repository}/pulls/{number}/reviews",
                "--input",
                "-",
                stdin=json.dumps(fallback),
            )
        return _json_object(raw, "create review")


def _comment_login(item: dict[str, Any]) -> str:
    user = item.get("user")
    if isinstance(user, dict):
        login = user.get("login")
        if isinstance(login, str):
            return login
    return "unknown"


def _json_object(raw: str, what: str) -> dict[str, Any]:
    _check_bytes(raw, MAX_PAYLOAD_BYTES, what)
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ActionError(f"{what} returned non-JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ActionError(f"{what} returned a non-object")
    return parsed


def _check_bytes(text: str, limit: int, label: str) -> None:
    if len(text) > limit or len(text.encode("utf-8")) > limit:
        raise InputLimitError(f"{label} exceeds the {limit} byte collection limit")


def _checked_diff(raw: str) -> str:
    _check_bytes(raw, MAX_DIFF_BYTES, "review diff")
    count = 0
    for _ in re.finditer(r"^diff --git ", raw, re.MULTILINE):
        count += 1
        if count > MAX_DIFF_FILES:
            raise InputLimitError(f"review diff exceeds the {MAX_DIFF_FILES} file collection limit")
    return raw


def _is_inline_comment_validation_error(message: str) -> bool:
    """Whether GitHub rejected inline placement, not the review request itself."""
    lowered = message.lower()
    return "422" in lowered and (
        "line is not part of the diff" in lowered
        or "validation failed" in lowered
        or "unprocessable entity" in lowered
    )


def require_full_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not _FULL_SHA_RE.fullmatch(value):
        raise ActionError(f"PR {label} SHA is missing or invalid; cannot collect the pinned diff")
    return value.lower()


class _CommandProtocol(asyncio.SubprocessProtocol):
    """Bound both output streams and expose completion after pipe closure/reap."""

    def __init__(self) -> None:
        self.outputs: list[list[bytes]] = [[], []]
        self.used = 0
        self.exceeded = False
        self.read_error: Exception | None = None
        self.done = asyncio.get_running_loop().create_future()
        self.closed = asyncio.get_running_loop().create_future()

    def connection_made(self, transport: asyncio.SubprocessTransport) -> None:
        self.transport = transport

    def pipe_data_received(self, fd: int, data: bytes) -> None:
        self.used += len(data)
        if self.used > MAX_COMMAND_BYTES:
            self.exceeded = True
            if not self.done.done():
                self.done.set_result(None)
            self.transport.close()
        elif not self.exceeded:
            self.outputs[fd - 1].append(data)

    def pipe_connection_lost(self, fd: int, exc: Exception | None) -> None:
        if exc is not None:
            self.read_error = exc
            if not self.done.done():
                self.done.set_result(None)
            self.transport.close()

    def connection_lost(self, exc: Exception | None) -> None:
        if exc is not None:
            self.read_error = exc
        if not self.done.done():
            self.done.set_result(None)
        self.closed.set_result(None)


async def _capture_command(
    cmd: list[str], *, timeout: int, label: str, **kwargs: Any
) -> tuple[int, bytes, bytes]:
    loop = asyncio.get_running_loop()
    transport, protocol = await loop.subprocess_exec(_CommandProtocol, *cmd, **kwargs)
    try:
        try:
            await asyncio.wait_for(asyncio.shield(protocol.done), timeout)
        except TimeoutError as exc:
            raise subprocess.TimeoutExpired(cmd, timeout) from exc
        if protocol.exceeded:
            raise InputLimitError(
                f"{label} output exceeds the {MAX_COMMAND_BYTES} byte collection limit"
            )
        if protocol.read_error is not None:
            raise ActionError(f"failed to read {label} output") from protocol.read_error
        return (
            transport.get_returncode(),
            b"".join(protocol.outputs[0]),
            b"".join(protocol.outputs[1]),
        )
    finally:
        # Public asyncio transports cancel pipe reads without joining a thread
        # blocked on EOF. Closing also kills a live child; wait for its reap and
        # pipe callbacks with a bounded cleanup window, even for inherited pipes.
        transport.close()
        await asyncio.wait_for(asyncio.shield(protocol.closed), 1)


def _bounded_command(
    cmd: list[str],
    *,
    timeout: int,
    label: str,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    stdin: str | None = None,
) -> tuple[int, bytes, bytes]:
    """Capture command output within byte and elapsed-time ceilings."""
    if stdin is not None:
        _check_bytes(stdin, MAX_PAYLOAD_BYTES, f"{label} request")
    # A bounded input file avoids a stdin/pipe deadlock while both output
    # streams are drained. No PR text is interpreted by a shell.
    with tempfile.TemporaryFile() as input_file:
        if stdin is not None:
            input_file.write(stdin.encode("utf-8"))
            input_file.seek(0)
        return asyncio.run(
            _capture_command(
                cmd,
                timeout=timeout,
                label=label,
                env=env,
                cwd=cwd,
                stdin=input_file,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )


def _run_gh(
    cmd: list[str],
    *,
    env: dict[str, str],
    timeout: int,
    stdin: str | None = None,
) -> str:
    try:
        returncode, stdout, stderr = _bounded_command(
            cmd,
            env=env,
            stdin=stdin,
            timeout=timeout,
            label="GitHub CLI",
        )
    except subprocess.TimeoutExpired as exc:
        raise ActionError(f"GitHub CLI timed out after {timeout}s") from exc
    except OSError as exc:
        raise ActionError(f"failed to run GitHub CLI: {exc}") from exc
    if returncode != 0:
        err = redact(stderr.decode("utf-8", errors="replace")[:600])
        raise ActionError(f"GitHub CLI failed ({' '.join(cmd[1:4])}): {err}")
    return stdout.decode("utf-8", errors="replace")


def _run_git(cmd: list[str], *, timeout: int, cwd: str) -> str:
    try:
        returncode, stdout, stderr = _bounded_command(
            cmd,
            cwd=cwd,
            timeout=timeout,
            label="Local git diff",
        )
    except subprocess.TimeoutExpired as exc:
        raise ActionError(f"Local git diff timed out after {timeout}s") from exc
    except OSError as exc:
        raise ActionError(f"failed to run local git diff: {exc}") from exc
    if returncode != 0:
        err = redact(stderr.decode("utf-8", errors="replace")[:600])
        raise ActionError(f"Local git diff failed: {err}")
    return stdout.decode("utf-8", errors="replace")


def upsert_status_comment(
    github: GitHub,
    *,
    pr_number: int,
    body: str,
    bot_login: str = "github-actions[bot]",
) -> str | None:
    text = f"{STATUS_MARKER}\n{body}".rstrip() + "\n"
    for comment in github.list_issue_comments(pr_number):
        existing = comment.get("body")
        if (
            _comment_login(comment) == bot_login
            and isinstance(existing, str)
            and STATUS_MARKER in existing
        ):
            comment_id = comment.get("id")
            if isinstance(comment_id, int):
                updated = github.update_issue_comment(comment_id, text)
                url = updated.get("html_url")
                return url if isinstance(url, str) else None
    created = github.create_issue_comment(pr_number, text)
    url = created.get("html_url")
    return url if isinstance(url, str) else None
