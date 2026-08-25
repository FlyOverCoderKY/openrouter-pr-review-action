"""Inert, read-only materialization of the reviewed commit.

PR code is never executed. Tools may only read files, search text, or list
directories inside this tree. No shell, no writes, no network.
"""

from __future__ import annotations

import io
import os
import re
import stat
import tarfile
from pathlib import Path

from or_pr_review.errors import ActionError, LaneError
from or_pr_review.redaction import looks_like_dotenv

MAX_MATERIALIZED_FILE = 1_000_000
# Per-read byte cap. The chat loop resends every tool observation on every
# later request, so large single reads multiply across the whole run; ranged
# reads (start_line/max_lines) fetch the rest when needed.
MAX_READ_BYTES = 60_000
MAX_RANGE_LINES = 2_000
DEFAULT_RANGE_LINES = 400
MAX_GREP_MATCHES = 100
MAX_LIST_ENTRIES = 200
MAX_TOOL_OUTPUT = 40_000

_BLOCKED_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "credentials.json",
    "service-account.json",
}
_BLOCKED_SUFFIXES = (".pem", ".p12", ".pfx", ".key")


def materialize_commit(repo: Path, sha: str, dest: Path) -> Path:
    """Extract tracked files for `sha` into dest. Symlinks and huge files skipped."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = _git_archive(repo, sha)
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            for member in tar.getmembers():
                if not _safe_member(member):
                    continue
                tar.extract(member, path=dest, filter="data")
    except (tarfile.TarError, OSError) as exc:
        raise ActionError(f"failed to materialize reviewed commit {sha[:12]}: {exc}") from exc
    return dest


def _git_archive(repo: Path, sha: str) -> bytes:
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", sha],
            capture_output=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ActionError(f"git archive failed: {exc}") from exc
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[:400]
        raise ActionError(
            f"git archive of {sha[:12]} failed (is the reviewed commit fetched? "
            f"use fetch-depth: 0). {err}"
        )
    return proc.stdout


def _safe_member(member: tarfile.TarInfo) -> bool:
    if not member.isfile():
        return False
    if member.size > MAX_MATERIALIZED_FILE:
        return False
    name = member.name.replace("\\", "/")
    parts = Path(name).parts
    if not parts or parts[0] in {"..", "/"} or ".." in parts:
        return False
    if name.startswith("/") or name.startswith("../"):
        return False
    mode = member.mode or 0
    if mode & stat.S_IXUSR:
        # Keep the file (reviewers may need to read scripts) but never execute.
        pass
    return True


def resolve_inside(root: Path, rel: str) -> Path:
    if not rel or rel.strip() in {".", "./"}:
        return root
    candidate = (root / rel).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise LaneError("path escapes the inert review workspace") from exc
    return candidate


def is_blocked_path(path: Path) -> bool:
    name = path.name
    if name in _BLOCKED_NAMES or name.startswith(".env."):
        return True
    return name.endswith(_BLOCKED_SUFFIXES)


def tool_read_file(
    root: Path,
    rel: str,
    start_line: int | None = None,
    max_lines: int | None = None,
) -> str:
    path = resolve_inside(root, rel)
    if not path.is_file():
        return f"error: not a file: {rel}"
    if is_blocked_path(path):
        return "error: refusing to read a secret-like path"
    try:
        data = path.read_bytes()
    except OSError as exc:
        return f"error: {exc}"
    text = data.decode("utf-8", errors="replace")
    if looks_like_dotenv(text):
        return "error: refusing to return .env-style KEY=value contents"
    if start_line is None and max_lines is None:
        if len(data) > MAX_READ_BYTES:
            clipped = data[:MAX_READ_BYTES].decode("utf-8", errors="replace")
            next_line = clipped.count("\n") + 1
            return clipped + (
                f"\n\n[truncated after {MAX_READ_BYTES} bytes; the file continues — "
                f"call read_file again with start_line={next_line} to keep reading]"
            )
        return text
    start = start_line if start_line is not None and start_line > 0 else 1
    count = max_lines if max_lines is not None and max_lines > 0 else DEFAULT_RANGE_LINES
    count = min(count, MAX_RANGE_LINES)
    lines = text.splitlines(keepends=True)
    total = len(lines)
    if total and start > total:
        return f"error: start_line {start} is past the end of the file ({total} lines)"
    window = lines[start - 1 : start - 1 + count]
    body = "".join(window)
    encoded = body.encode("utf-8")
    truncated_bytes = len(encoded) > MAX_READ_BYTES
    if truncated_bytes:
        body = encoded[:MAX_READ_BYTES].decode("utf-8", errors="replace")
    end = min(start + len(window) - 1, total)
    header = f"[lines {start}-{end} of {total}]\n"
    if truncated_bytes:
        footer = f"\n[window truncated after {MAX_READ_BYTES} bytes]"
    elif end < total:
        footer = f"\n[file continues; call read_file with start_line={end + 1} for more]"
    else:
        footer = ""
    return header + body + footer


def tool_list_dir(root: Path, rel: str) -> str:
    path = resolve_inside(root, rel)
    if not path.is_dir():
        return f"error: not a directory: {rel}"
    try:
        entries = sorted(path.iterdir(), key=lambda item: item.name.lower())
    except OSError as exc:
        return f"error: {exc}"
    lines: list[str] = []
    for entry in entries[:MAX_LIST_ENTRIES]:
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"{entry.name}{suffix}")
    if len(entries) > MAX_LIST_ENTRIES:
        lines.append(f"[{len(entries) - MAX_LIST_ENTRIES} more entries omitted]")
    return "\n".join(lines) if lines else "(empty)"


def tool_grep(root: Path, pattern: str, rel: str = ".") -> str:
    if not pattern:
        return "error: pattern is required"
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"error: invalid pattern: {exc}"
    start = resolve_inside(root, rel)
    if not start.exists():
        return f"error: path not found: {rel}"
    matches: list[str] = []
    files = [start] if start.is_file() else _walk_files(start)
    for path in files:
        if is_blocked_path(path):
            continue
        try:
            if path.stat().st_size > MAX_MATERIALIZED_FILE:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if looks_like_dotenv(text):
            continue
        rel_path = path.relative_to(root.resolve()).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append(f"{rel_path}:{lineno}:{line[:400]}")
                if len(matches) >= MAX_GREP_MATCHES:
                    matches.append("[grep match cap reached]")
                    return _cap_output("\n".join(matches))
    return _cap_output("\n".join(matches) if matches else "no matches")


def _walk_files(start: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(start):
        dirnames[:] = [name for name in dirnames if name not in {".git", ".hg"}]
        for name in filenames:
            out.append(Path(dirpath) / name)
            if len(out) > 5_000:
                return out
    return out


def _cap_output(text: str) -> str:
    data = text.encode("utf-8")
    if len(data) <= MAX_TOOL_OUTPUT:
        return text
    return data[:MAX_TOOL_OUTPUT].decode("utf-8", errors="ignore") + "\n[tool output truncated]"


READ_ONLY_TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a tracked file from the inert checkout of the reviewed commit. "
                "Read-only. Paths are relative to the repository root. Use this for "
                "README, code-map docs, sibling workflows, and tests that inventory "
                "filenames — not only files named in the embedded diff. Large files "
                "are truncated; pass start_line (and optionally max_lines) to read "
                "a specific window instead of the whole file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repository-relative path"},
                    "start_line": {
                        "type": "integer",
                        "description": "Optional 1-based first line for a ranged read",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Optional line count for a ranged read (default 400)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search file contents in the inert checkout with a Python regular expression. "
                "Read-only. Does not run a shell. Use this to find tests, docs, or configs "
                "that list workflow filenames or other inventories of the changed paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {
                        "type": "string",
                        "description": "Optional file or directory to search (default: repo root)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "List a directory in the inert checkout. Read-only. Use this for "
                "sibling CI files (for example .github/workflows) and test/docs trees."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory relative to the repository root (default: .)",
                    }
                },
                "required": [],
            },
        },
    },
)


def dispatch_tool(root: Path, name: str, arguments: dict[str, object]) -> str:
    try:
        return _dispatch_tool(root, name, arguments)
    except LaneError as exc:
        return f"error: {exc}"


def _optional_int_arg(value: object) -> tuple[int | None, bool]:
    """Return (parsed, ok). None with ok=True means the argument was absent."""
    if value is None:
        return None, True
    if isinstance(value, bool):
        return None, False
    if isinstance(value, int):
        return value, True
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip()), True
    return None, False


def _dispatch_tool(root: Path, name: str, arguments: dict[str, object]) -> str:
    if name == "read_file":
        path = arguments.get("path")
        if not isinstance(path, str):
            return "error: path is required"
        start_line, ok = _optional_int_arg(arguments.get("start_line"))
        if not ok:
            return "error: start_line must be an integer"
        max_lines, ok = _optional_int_arg(arguments.get("max_lines"))
        if not ok:
            return "error: max_lines must be an integer"
        return tool_read_file(root, path, start_line=start_line, max_lines=max_lines)
    if name == "grep":
        pattern = arguments.get("pattern")
        if not isinstance(pattern, str):
            return "error: pattern is required"
        path = arguments.get("path", ".")
        if not isinstance(path, str):
            path = "."
        return tool_grep(root, pattern, path)
    if name == "list_dir":
        path = arguments.get("path", ".")
        if not isinstance(path, str):
            path = "."
        return tool_list_dir(root, path)
    return f"error: unknown or disallowed tool {name!r}"
