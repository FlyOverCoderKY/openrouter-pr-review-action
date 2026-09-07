"""Bounded, base-commit-only discovery for hierarchical ``REVIEW.md`` policy.

``resolve_policy`` is deliberately a Git plumbing reader: it never consults the
working tree, executes repository code, or follows filesystem links.  Its
``base_sha`` is the caller-selected trusted target tip; the three-dot diff is
used only to decide which paths need policy coverage.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from io import BufferedReader
from pathlib import Path
from typing import Any

from or_pr_review.errors import ActionError
from or_pr_review.redaction import redact
from or_pr_review.schema import MAX_FILE, normalize_review_path
from or_pr_review.triage import path_glob_regex

_SHA = re.compile(r"^[0-9a-f]{40}$")
_PROFILE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_FILE_BYTES = 16 * 1024
_TOTAL_BYTES = 64 * 1024
_MAX_FILES = 32
_MAX_DEPTH = 20
_MAX_RULES = 64
_MAX_GLOBS_RULE = 16
_MAX_GLOBS = 256
_MAX_GLOB_BYTES = 256
_GIT_OUTPUT = 4 * 1024 * 1024


@dataclass(frozen=True)
class PolicyFile:
    path: str
    blob_sha: str
    content: str
    scope_paths: tuple[str, ...]


@dataclass(frozen=True)
class PolicyRuleMatch:
    file: str
    id: str
    paths: tuple[str, ...]
    minimum: str


@dataclass(frozen=True)
class ResolvedPolicy:
    base_sha: str
    profile: str
    minimum: str
    files: tuple[PolicyFile, ...]
    matches: tuple[PolicyRuleMatch, ...]
    changed_paths: tuple[str, ...]
    digest: str


@dataclass(frozen=True)
class _Rule:
    id: str
    paths: tuple[str, ...]
    minimum: str


@dataclass(frozen=True)
class _Parsed:
    prose: str
    profile: str | None
    minimum: str | None
    rules: tuple[_Rule, ...]
    metadata: dict[str, Any]


def _error(path: str, reason: str) -> ActionError:
    return ActionError(f"{path}: {reason}")


def _clean_env() -> dict[str, str]:
    """Return inherited process essentials without credentials or Git overrides."""
    banned = ("TOKEN", "SECRET", "API_KEY", "OPENROUTER", "GITHUB", "GH_")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_") and not any(token in key.upper() for token in banned)
    }
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    return env


_GIT_STDERR = 8192


def _drain_bounded(
    pipe: BufferedReader | None,
    limit: int,
    overflow: threading.Event,
    read_error: threading.Event,
    read_error_detail: list[BaseException],
) -> bytes:
    if pipe is None:
        return b""
    stored = bytearray()
    try:
        while not overflow.is_set():
            remaining = limit - len(stored)
            request = min(65536, remaining + 1)
            try:
                chunk = pipe.read1(request)
            except OSError as exc:
                read_error.set()
                read_error_detail.append(exc)
                break
            if not chunk:
                break
            stored.extend(chunk)
            if len(stored) > limit:
                overflow.set()
                del stored[limit:]
                break
    finally:
        try:
            pipe.close()
        except OSError:
            pass
    return bytes(stored)


def _git_operation(args: list[str]) -> str:
    if not args:
        return "command"
    verb = args[0]
    if verb == "cat-file" and len(args) >= 2:
        return f"{verb} {args[1]}"
    return verb


def _git(repo: Path, args: list[str], timeout: int, limit: int = _GIT_OUTPUT) -> bytes:
    command = [
        "git",
        "--no-replace-objects",
        "--no-lazy-fetch",
        "-c",
        "core.quotepath=false",
        "-C",
        str(repo),
        *args,
    ]
    try:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_clean_env(),
        )
    except OSError as exc:
        raise ActionError(f"failed to run git policy discovery: {exc}") from exc

    stdout_box: list[bytes] = []
    stderr_box: list[bytes] = []
    stdout_overflow = threading.Event()
    stderr_overflow = threading.Event()
    stdout_read_error = threading.Event()
    stderr_read_error = threading.Event()
    stdout_read_error_detail: list[BaseException] = []
    stderr_read_error_detail: list[BaseException] = []

    def read_stdout() -> None:
        stdout_box.append(
            _drain_bounded(
                proc.stdout,
                limit,
                stdout_overflow,
                stdout_read_error,
                stdout_read_error_detail,
            )
        )

    def read_stderr() -> None:
        stderr_box.append(
            _drain_bounded(
                proc.stderr,
                _GIT_STDERR,
                stderr_overflow,
                stderr_read_error,
                stderr_read_error_detail,
            )
        )

    threads = (
        threading.Thread(target=read_stdout, daemon=True),
        threading.Thread(target=read_stderr, daemon=True),
    )
    for thread in threads:
        thread.start()

    deadline = time.monotonic() + timeout
    timed_out = False
    try:
        while proc.poll() is None:
            if (
                stdout_overflow.is_set()
                or stderr_overflow.is_set()
                or stdout_read_error.is_set()
                or stderr_read_error.is_set()
            ):
                proc.kill()
                break
            if time.monotonic() >= deadline:
                timed_out = True
                proc.kill()
                break
            time.sleep(0.01)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    finally:
        for thread in threads:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in threads):
            raise ActionError("git policy discovery reader did not finish")

    if timed_out:
        raise ActionError(f"git policy discovery timed out after {timeout}s")
    if stdout_read_error.is_set() or stderr_read_error.is_set():
        raise ActionError("git policy discovery output read failed")
    if stdout_overflow.is_set() or stderr_overflow.is_set():
        raise ActionError("git policy discovery output exceeds its safety bound")
    if proc.returncode:
        stderr_text = stderr_box[0] if stderr_box else b""
        detail = redact(stderr_text.decode("utf-8", errors="replace").strip())
        if len(detail) > 600:
            detail = detail[:600]
        operation = _git_operation(args)
        message = f"git policy discovery {operation} failed"
        if detail:
            message += f": {detail}"
        raise ActionError(message)
    return stdout_box[0] if stdout_box else b""


def _validate_commit(repo: Path, sha: str, timeout: int, label: str) -> None:
    if not isinstance(sha, str) or not _SHA.fullmatch(sha):
        raise ActionError(f"{label}_sha must be a full lowercase 40-character hex SHA")
    kind = _git(repo, ["cat-file", "-t", sha], timeout, 128).decode("ascii", "replace").strip()
    if kind != "commit":
        raise ActionError(f"{label}_sha is not a commit object")


def _discovery_path(path: str, label: str) -> str:
    normalized = normalize_review_path(path)
    if normalized is None:
        if len(path) > MAX_FILE:
            raise ActionError(f"{label} path exceeds {MAX_FILE} characters")
        if "`" in path:
            raise ActionError(f"invalid {label} path {path!r}: backticks are not allowed")
        raise ActionError(f"invalid {label} path {path!r}")
    if path != normalized:
        raise ActionError(f"invalid {label} path {path!r}: path is not canonical")
    if ":" in path:
        raise ActionError(f"invalid {label} path {path!r}: colons are not allowed")
    return path


def _safe_path(path: str, label: str) -> str:
    """Glob-segment validation; review paths use ``_discovery_path`` instead."""
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or ":" in path
        or any(ord(char) < 32 or ord(char) == 127 for char in path)
    ):
        raise ActionError(f"unsafe {label} path {path!r}")
    bits = path.split("/")
    if any(bit in {"", ".", ".."} for bit in bits):
        raise ActionError(f"unsafe {label} path {path!r}")
    return path


def _ls_tree_children(repo: Path, tree_oid: str, timeout: int) -> list[tuple[str, str, str, str]]:
    if not _SHA.fullmatch(tree_oid):
        raise ActionError("internal tree object id is invalid")
    raw = _git(repo, ["ls-tree", "--full-tree", "-z", tree_oid], timeout)
    children: list[tuple[str, str, str, str]] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            left, name = item.split(b"\t", 1)
            mode, kind, object_sha = left.decode("ascii").split(" ")
        except ValueError as exc:
            raise ActionError("tree listing contains an invalid entry") from exc
        try:
            name_str = name.decode("utf-8", "strict")
        except UnicodeDecodeError:
            continue
        children.append((name_str, mode, kind, object_sha))
    return children


def _discover_entries(
    repo: Path,
    commit_sha: str,
    directories: set[str],
    timeout: int,
) -> dict[str, tuple[str, str, str]]:
    """List immediate children of the root and applicable ancestor directories."""
    root_tree = (
        _git(repo, ["rev-parse", "--verify", f"{commit_sha}^{{tree}}"], timeout, 128)
        .decode("ascii", "replace")
        .strip()
    )
    if not _SHA.fullmatch(root_tree):
        raise ActionError("base commit tree is invalid")

    tree_oids: dict[str, str] = {"": root_tree}
    entries: dict[str, tuple[str, str, str]] = {}
    for directory in sorted(
        directories, key=lambda item: (0 if not item else item.count("/") + 1, item)
    ):
        tree_oid = tree_oids.get(directory)
        if tree_oid is None:
            continue
        for name, mode, kind, object_sha in _ls_tree_children(repo, tree_oid, timeout):
            full_path = f"{directory}/{name}" if directory else name
            if name.casefold() == "review.md":
                # Policy names are always applicable when their directory is
                # being enumerated: validate bounds even for the exact name.
                _discovery_path(full_path, "base tree")
                if name != "REVIEW.md":
                    raise _error(full_path, "case-variant REVIEW.md is ambiguous")
            try:
                full_path = _discovery_path(full_path, "base tree")
            except ActionError:
                continue
            entries[full_path] = (mode, kind, object_sha)
            if kind == "tree" and full_path in directories:
                tree_oids[full_path] = object_sha
    return entries


def _changed_paths(repo: Path, base: str, head: str, timeout: int) -> tuple[str, ...]:
    raw = _git(
        repo,
        [
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--find-renames",
            "--no-relative",
            "--name-status",
            "-z",
            f"{base}...{head}",
        ],
        timeout,
    )
    items = raw.split(b"\0")
    paths: set[str] = set()
    index = 0
    while index < len(items) - 1:
        status = items[index].decode("ascii", "replace")
        index += 1
        if not status:
            continue
        count = 2 if status[:1] in {"R", "C"} else 1
        if index + count > len(items):
            raise ActionError("git name-status output is malformed")
        for raw_path in items[index : index + count]:
            try:
                paths.add(_discovery_path(raw_path.decode("utf-8", "strict"), "changed"))
            except UnicodeDecodeError as exc:
                raise ActionError("git name-status includes a non-UTF-8 path") from exc
        index += count
    return tuple(sorted(paths))


def _parse_json(text: str, path: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _error(path, f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=no_duplicates,
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
        )
    except (json.JSONDecodeError, UnicodeError, RecursionError, ValueError) as exc:
        raise _error(path, f"invalid review-policy JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise _error(path, "review-policy metadata must be an object")
    _reject_surrogates(value, path)
    return value


def _reject_surrogates(value: Any, path: str) -> None:
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise _error(path, "review-policy metadata contains an invalid Unicode surrogate")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_surrogates(key, path)
            _reject_surrogates(item, path)
    elif isinstance(value, list):
        for item in value:
            _reject_surrogates(item, path)


def _valid_glob(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > _MAX_GLOB_BYTES:
        raise _error(path, "rule paths must be non-empty relative globs of at most 256 UTF-8 bytes")
    try:
        _safe_path(value, "rule glob")
        path_glob_regex(value)
    except (re.error, ActionError) as exc:
        raise _error(path, f"invalid rule glob {value!r}") from exc
    return value


def _parse_policy(path: str, raw: bytes, *, root: bool) -> _Parsed:
    if len(raw) > _FILE_BYTES:
        raise _error(path, "REVIEW.md exceeds 16 KiB")
    try:
        text = raw.decode("utf-8-sig", "strict")
    except UnicodeDecodeError as exc:
        raise _error(path, "REVIEW.md is not valid UTF-8") from exc
    # A metadata fence is accepted only at byte/text beginning (after optional BOM).
    metadata: dict[str, Any] = {"version": 1, "review": {}}
    prose = text
    if text.startswith("```review-policy\n") or text.startswith("```review-policy\r\n"):
        first_end = text.find("\n") + 1
        close = re.search(r"(?m)^```[ \t]*(?:\r?$)", text[first_end:])
        if not close:
            raise _error(path, "review-policy metadata fence is unclosed")
        start = first_end + close.start()
        end = first_end + close.end()
        metadata = _parse_json(text[first_end:start], path)
        prose = text[end:]
    misplaced = not text.startswith("```review-policy") and "```review-policy" in text
    if "```review-policy" in prose or misplaced:
        raise _error(path, "review-policy metadata fence must be the first content")
    if (
        set(metadata) != {"version", "review"}
        or type(metadata["version"]) is not int
        or metadata["version"] != 1
        or not isinstance(metadata["review"], dict)
    ):
        raise _error(path, "metadata must be exactly {'version': 1, 'review': {...}}")
    review = metadata["review"]
    allowed = {"profile", "minimum", "rules"}
    if set(review) - allowed:
        raise _error(path, "review metadata contains an unknown key")
    profile = review.get("profile")
    if "profile" in review:
        if not root:
            raise _error(path, "only root REVIEW.md may set profile")
        if not isinstance(profile, str) or not _PROFILE.fullmatch(profile):
            raise _error(path, "profile must be a non-empty bounded slug")
    minimum = review.get("minimum")
    if "minimum" in review and (
        not isinstance(minimum, str) or minimum not in {"standard", "deep"}
    ):
        raise _error(path, "minimum must be standard or deep")
    raw_rules = review.get("rules", [])
    if not isinstance(raw_rules, list):
        raise _error(path, "rules must be an array")
    rules: list[_Rule] = []
    ids: set[str] = set()
    for rule in raw_rules:
        if not isinstance(rule, dict) or set(rule) != {"id", "paths", "minimum"}:
            raise _error(path, "each rule must contain exactly id, paths, and minimum")
        rule_id = rule["id"]
        if (
            not isinstance(rule_id, str)
            or not rule_id
            or len(rule_id) > 128
            or any(ord(char) < 32 or ord(char) == 127 for char in rule_id)
        ):
            raise _error(path, "rule id must be a non-empty string of at most 128 characters")
        if rule_id in ids:
            raise _error(path, f"duplicate rule id {rule_id!r}")
        ids.add(rule_id)
        raw_paths = rule["paths"]
        if not isinstance(raw_paths, list) or not raw_paths or len(raw_paths) > _MAX_GLOBS_RULE:
            raise _error(path, "rule paths must contain 1 through 16 globs")
        if not isinstance(rule["minimum"], str) or rule["minimum"] not in {"standard", "deep"}:
            raise _error(path, "rule minimum must be standard or deep")
        rules.append(
            _Rule(rule_id, tuple(_valid_glob(x, path) for x in raw_paths), rule["minimum"])
        )
    return _Parsed(prose, profile, minimum, tuple(rules), metadata)


def parse_policy_file(path: str, raw: bytes) -> _Parsed:
    """Parse one REVIEW.md blob for local syntax validation."""
    if not isinstance(path, str):
        raise ActionError("policy path must be a string")
    if not isinstance(raw, bytes):
        raise ActionError("policy content must be bytes")
    return _parse_policy(path, raw, root=path == "REVIEW.md")


def resolve_policy(
    repo: Path,
    base_sha: str,
    head_sha: str,
    carried_paths: tuple[str, ...] = (),
    timeout: int = 120,
) -> ResolvedPolicy:
    """Resolve trusted-base REVIEW.md files for all changed and carried paths.

    ``carried_paths`` are treated like changed paths so prior findings retain
    their applicable policy even when an embedded provider diff is truncated.
    """
    if not isinstance(timeout, int) or timeout < 1:
        raise ActionError("timeout must be a positive integer")
    _validate_commit(repo, base_sha, timeout, "base")
    _validate_commit(repo, head_sha, timeout, "head")
    changed = set(_changed_paths(repo, base_sha, head_sha, timeout))
    for item in carried_paths:
        if not isinstance(item, str):
            raise ActionError("carried_paths must contain strings")
        changed.add(_discovery_path(item, "carried"))
    changed_paths = tuple(sorted(changed))
    directories: set[str] = {""}
    for changed_path in changed_paths:
        parts = changed_path.split("/")
        if len(parts) > _MAX_DEPTH:
            raise ActionError(f"{changed_path}: ancestor depth exceeds {_MAX_DEPTH}")
        for index in range(len(parts)):
            directories.add("/".join(parts[:index]))

    tree = _discover_entries(repo, base_sha, directories, timeout)

    for changed_path in changed_paths:
        parts = changed_path.split("/")
        for index in range(len(parts)):
            component = "/".join(parts[: index + 1])
            entry = tree.get(component)
            if entry and entry[0] == "120000":
                raise ActionError(f"{changed_path}: traverses symlink {component}")
            if entry and entry[0] == "160000":
                raise ActionError(f"{changed_path}: traverses submodule {component}")

    policy_paths = [
        (directory + "/" if directory else "") + "REVIEW.md"
        for directory in directories
        if ((directory + "/" if directory else "") + "REVIEW.md") in tree
    ]
    policy_paths.sort(key=lambda x: (x.count("/"), x))
    if len(policy_paths) > _MAX_FILES:
        raise ActionError("applicable REVIEW.md file count exceeds 32")

    files: list[PolicyFile] = []
    parsed_by_path: dict[str, _Parsed] = {}
    total = 0
    all_rules = 0
    all_globs = 0
    for policy_path in policy_paths:
        mode, kind, blob_sha = tree[policy_path]
        if mode not in {"100644", "100755"} or kind != "blob":
            raise _error(
                policy_path,
                "REVIEW.md must be a regular blob (not a symlink or submodule)",
            )
        size = int(_git(repo, ["cat-file", "-s", blob_sha], timeout, 128))
        if size > _FILE_BYTES:
            raise _error(policy_path, "REVIEW.md exceeds 16 KiB")
        if total + size > _TOTAL_BYTES:
            raise _error(policy_path, "applicable REVIEW.md aggregate bytes exceed 64 KiB")
        raw = _git(repo, ["cat-file", "blob", blob_sha], timeout, _FILE_BYTES)
        total += len(raw)
        parsed = _parse_policy(policy_path, raw, root=policy_path == "REVIEW.md")
        all_rules += len(parsed.rules)
        all_globs += sum(len(rule.paths) for rule in parsed.rules)
        if all_rules > _MAX_RULES or all_globs > _MAX_GLOBS:
            raise _error(policy_path, "applicable rules or globs exceed configured safety bound")
        directory = policy_path.rsplit("/", 1)[0] if "/" in policy_path else ""
        prefix = directory + "/" if directory else ""
        scope = tuple(path for path in changed_paths if path.startswith(prefix))
        files.append(PolicyFile(policy_path, blob_sha, parsed.prose, scope))
        parsed_by_path[policy_path] = parsed

    profile = "code"
    minimum = "standard"
    matches: list[PolicyRuleMatch] = []
    for file in files:
        parsed = parsed_by_path[file.path]
        if parsed.profile is not None:
            profile = parsed.profile
        if parsed.minimum == "deep":
            minimum = "deep"
        directory = file.path.rsplit("/", 1)[0] if "/" in file.path else ""
        prefix = directory + "/" if directory else ""
        for rule in parsed.rules:
            regexes = tuple(path_glob_regex(pattern) for pattern in rule.paths)
            hit = tuple(
                path
                for path in file.scope_paths
                if any(regex.fullmatch(path[len(prefix) :]) for regex in regexes)
            )
            if hit:
                matches.append(
                    PolicyRuleMatch(file.path, f"{file.path}:{rule.id}", hit, rule.minimum)
                )
                if rule.minimum == "deep":
                    minimum = "deep"
    payload = {
        "profile": profile,
        "minimum": minimum,
        "changed_paths": changed_paths,
        "files": [
            (f.path, f.blob_sha, f.scope_paths, parsed_by_path[f.path].metadata) for f in files
        ],
        "matches": [(m.file, m.id, m.paths, m.minimum) for m in matches],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ResolvedPolicy(
        base_sha, profile, minimum, tuple(files), tuple(matches), changed_paths, digest
    )
