"""Integration coverage for REVIEW.md discovery (each fixture is a real Git repo)."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from or_pr_review.errors import ActionError
from or_pr_review.review_policy import (
    _GIT_OUTPUT,
    _GIT_STDERR,
    _discover_entries,
    _discovery_path,
    parse_policy_file,
    resolve_policy,
)
from or_pr_review.review_policy import (
    _git as bounded_git,
)


def _git(repo: Path, *args: str, input: str | bytes | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=not isinstance(input, bytes),
        input=input,
        capture_output=True,
    )
    return (
        result.stdout.decode().strip()
        if isinstance(result.stdout, bytes)
        else result.stdout.strip()
    )


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    _git(tmp_path, "init", "repo")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    return repo


def _commit(repo: Path, files: dict[str, str], message: str = "commit") -> str:
    for name, content in files.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _policy(review: str) -> str:
    return "```review-policy\n{" + '"version":1,"review":' + review + "}\n```\nInstructions.\n"


def test_trusted_base_hierarchy_scope_and_globs(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = _commit(
        repo,
        {
            "REVIEW.md": _policy(
                '{"profile":"security","minimum":"standard","rules":['
                '{"id":"py","paths":["src/**/*.py"],"minimum":"deep"}]}'
            ),
            "src/REVIEW.md": _policy(
                '{"minimum":"deep","rules":[{"id":"local","paths":["*.py"],"minimum":"standard"}]}'
            ),
            "docs/REVIEW.md": "Docs policy.\n",
            "src/a.py": "a\n",
            "docs/a.md": "a\n",
        },
    )
    head = _commit(repo, {"src/a.py": "changed\n", "docs/a.md": "changed\n"})
    found = resolve_policy(repo, base, head)
    assert found.profile == "security" and found.minimum == "deep"
    assert [x.path for x in found.files] == ["REVIEW.md", "docs/REVIEW.md", "src/REVIEW.md"]
    assert [(x.id, x.paths) for x in found.matches] == [
        ("REVIEW.md:py", ("src/a.py",)),
        ("src/REVIEW.md:local", ("src/a.py",)),
    ]
    assert found.files[0].content == "\nInstructions.\n"


def test_changed_head_policy_is_not_trusted_and_digest_is_deterministic(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = _commit(repo, {"a.py": "a\n"})
    head = _commit(
        repo,
        {"REVIEW.md": _policy('{"profile":"malicious","minimum":"deep"}'), "a.py": "b\n"},
    )
    one = resolve_policy(repo, base, head)
    two = resolve_policy(repo, base, head)
    assert (one.profile, one.minimum, one.files, one.digest) == ("code", "standard", (), two.digest)


def test_rename_deletion_and_carried_paths_are_covered(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = _commit(
        repo,
        {
            "old/REVIEW.md": "old\n",
            "gone/REVIEW.md": "gone\n",
            "old/a.py": "a\n",
            "gone/b.py": "b\n",
            "carry/REVIEW.md": "carry\n",
            "carry/c.py": "c\n",
        },
    )
    _git(repo, "mv", "old/a.py", "new.py")
    (repo / "gone/b.py").unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "rename and delete")
    head = _git(repo, "rev-parse", "HEAD")
    found = resolve_policy(repo, base, head, ("carry/c.py",))
    assert set(found.changed_paths) == {"old/a.py", "new.py", "gone/b.py", "carry/c.py"}
    assert {x.path for x in found.files} == {"old/REVIEW.md", "gone/REVIEW.md", "carry/REVIEW.md"}


@pytest.mark.parametrize(
    "contents, match",
    [
        ("```review-policy\n{}", "unclosed"),
        ("text\n```review-policy\n{}\n```\n", "first content"),
        (_policy('{"profile":"x","extra":true}'), "unknown key"),
        (
            _policy('{"rules":[{"id":"x","id":"y","paths":["a"],"minimum":"deep"}]}'),
            "duplicate JSON key",
        ),
        ("```review-policy \n{}\n```\n", "first content"),
        ("```review-policy\t\n{}\n```\n", "first content"),
    ],
)
def test_malformed_metadata_is_rejected(tmp_path: Path, contents: str, match: str) -> None:
    repo = _repo(tmp_path)
    base = _commit(repo, {"REVIEW.md": contents, "a": "a"})
    head = _commit(repo, {"a": "b"})
    with pytest.raises(ActionError, match=match):
        resolve_policy(repo, base, head)


def test_bounds_case_ambiguity_and_object_validation(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = _commit(repo, {"review.md": "wrong case\n", "a": "a"})
    head = _commit(repo, {"a": "b"})
    with pytest.raises(ActionError, match="case-variant"):
        resolve_policy(repo, base, head)
    with pytest.raises(ActionError, match="full lowercase"):
        resolve_policy(repo, "A" * 40, head)
    with pytest.raises(ActionError, match="not a commit"):
        resolve_policy(repo, _git(repo, "hash-object", "a"), head)


def test_empty_diff_root_case_variant_is_rejected(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = _commit(repo, {"reView.md": "wrong case\n", "a": "a"})
    with pytest.raises(ActionError, match="case-variant"):
        resolve_policy(repo, base, base)


_GIT_CHILD = """\
import sys
import time

mode = sys.argv[1]
limit = int(sys.argv[2])
if mode == "stdout_overflow":
    sys.stdout.buffer.write(b"x" * (limit + 1))
    sys.stdout.buffer.flush()
    time.sleep(30)
elif mode == "stderr_overflow":
    sys.stderr.buffer.write(b"x" * (limit + 1))
    sys.stderr.buffer.flush()
    time.sleep(30)
elif mode == "stall":
    time.sleep(3600)
elif mode == "ok":
    sys.stdout.buffer.write(b"commit\\n")
elif mode == "fail_stderr":
    secret = "ghp_supersecrettokenvalue123456"
    sys.stderr.buffer.write(f"fatal: missing object {secret}\\n".encode())
    sys.stderr.buffer.flush()
    sys.exit(1)
"""


def _patch_git_popen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, limit: int = 0
) -> None:
    script = tmp_path / "git_child.py"
    script.write_text(_GIT_CHILD, encoding="utf-8")
    real_popen = subprocess.Popen

    def fake_popen(cmd, **kwargs):
        if isinstance(cmd, (list, tuple)) and cmd and cmd[0] == "git":
            return real_popen([sys.executable, "-I", str(script), mode, str(limit)], **kwargs)
        return real_popen(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)


def test_git_output_bounds_kill_overflowing_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    base = _commit(repo, {"a": "a"})
    limit = 64
    _patch_git_popen(monkeypatch, tmp_path, "stdout_overflow", limit)
    start = time.monotonic()
    with pytest.raises(ActionError, match="safety bound"):
        bounded_git(repo, ["cat-file", "-t", base], 10, limit)
    assert time.monotonic() - start < 3


def test_git_output_bounds_kill_overflowing_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    base = _commit(repo, {"a": "a"})
    _patch_git_popen(monkeypatch, tmp_path, "stderr_overflow", _GIT_STDERR)
    start = time.monotonic()
    with pytest.raises(ActionError, match="safety bound"):
        bounded_git(repo, ["cat-file", "-t", base], 10, _GIT_OUTPUT)
    assert time.monotonic() - start < 3


def test_git_output_bounds_kill_stalled_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    base = _commit(repo, {"a": "a"})
    _patch_git_popen(monkeypatch, tmp_path, "stall")
    with pytest.raises(ActionError, match="timed out"):
        bounded_git(repo, ["cat-file", "-t", base], 1, _GIT_OUTPUT)


def test_byte_and_file_count_limits(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    files = {"a": "a", "REVIEW.md": "x" * (16 * 1024 + 1)}
    base = _commit(repo, files)
    head = _commit(repo, {"a": "b"})
    with pytest.raises(ActionError, match="16 KiB"):
        resolve_policy(repo, base, head)
    # A separate history gives every changed sibling its own applicable policy.
    second = tmp_path / "second"
    second.mkdir()
    repo = _repo(second)
    files = {f"d{i}/REVIEW.md": "x" for i in range(33)} | {f"d{i}/a": "a" for i in range(33)}
    base = _commit(repo, files)
    head = _commit(repo, {f"d{i}/a": "b" for i in range(33)})
    with pytest.raises(ActionError, match="count exceeds 32"):
        resolve_policy(repo, base, head)


def test_aggregate_byte_limit_and_unrelated_policy_are_bounded(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    files = {"bad/REVIEW.md": "```review-policy\nnot-json\n```"}
    files.update({f"d{i}/REVIEW.md": "x" * (16 * 1024) for i in range(5)})
    files.update({f"d{i}/a": "a" for i in range(5)})
    files["src/a"] = "a"
    base = _commit(repo, files)
    unrelated_head = _commit(repo, {"src/a": "b"})
    assert resolve_policy(repo, base, unrelated_head).files == ()
    head = _commit(repo, {f"d{i}/a": "b" for i in range(5)})
    with pytest.raises(ActionError, match="aggregate bytes exceed 64 KiB"):
        resolve_policy(repo, base, head)


@pytest.mark.parametrize(
    "path, raw, match",
    [
        ("REVIEW.md", b'```review-policy\n{"version":true,"review":{}}\n```', "metadata"),
        ("REVIEW.md", b'```review-policy\n{"version":1.0,"review":{}}\n```', "metadata"),
        (
            "REVIEW.md",
            b'{"version":1,"review":{"rules":[{"id":"x","paths":["a"],"minimum":[]}]}}',
            "minimum",
        ),
        (
            "REVIEW.md",
            b'{"version":1,"review":{"rules":[{"id":"x\\n","paths":["a"],"minimum":"deep"}]}}',
            "rule id",
        ),
        (
            "REVIEW.md",
            b'{"version":1,"review":{"rules":[{"id":"x","paths":["C:/x"],"minimum":"deep"}]}}',
            "invalid rule glob",
        ),
        (
            "REVIEW.md",
            b'{"version":1,"review":{"rules":[{"id":"x","paths":["a\\tb"],"minimum":"deep"}]}}',
            "invalid rule glob",
        ),
        ("REVIEW.md", b'{"version":NaN,"review":{}}', "invalid review-policy JSON"),
        ("REVIEW.md", b'{"version":2,"review":{}}', "metadata"),
        ("REVIEW.md", b'{"version":1,"review":{"unknown":true}}', "unknown key"),
        ("REVIEW.md", b'{"version":1,"review":{"minimum":"\\ud800"}}', "surrogate"),
    ],
)
def test_parse_policy_file_rejects_invalid_metadata(path: str, raw: bytes, match: str) -> None:
    fenced = raw
    if not raw.startswith(b"```"):
        fenced = b"```review-policy" + bytes((10,)) + raw + bytes((10,)) + b"```"
    with pytest.raises(ActionError, match=match):
        parse_policy_file(path, fenced)


def test_parse_policy_file_enforces_root_only_profile() -> None:
    raw = _policy('{"profile":"security"}').encode()
    assert parse_policy_file("REVIEW.md", raw).profile == "security"
    with pytest.raises(ActionError, match="only root"):
        parse_policy_file("src/REVIEW.md", raw)


def test_empty_diff_uses_root_guidance_and_glob_depth_is_exact(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = _commit(
        repo,
        {
            "REVIEW.md": _policy(
                '{"minimum":"deep","rules":['
                '{"id":"flat","paths":["src/*.py"],"minimum":"deep"},'
                '{"id":"deep","paths":["src/**/*.py"],"minimum":"deep"}]}'
            ),
            "src/a.py": "a",
            "src/nested/a.py": "a",
        },
    )
    empty = resolve_policy(repo, base, base)
    assert (empty.minimum, [file.path for file in empty.files]) == ("deep", ["REVIEW.md"])
    head = _commit(repo, {"src/nested/a.py": "b"})
    found = resolve_policy(repo, base, head)
    assert [match.id for match in found.matches] == ["REVIEW.md:deep"]


def test_rule_glob_and_depth_caps_and_nested_standard_do_not_lower_root(tmp_path: Path) -> None:
    too_many_rules = ",".join(
        f'{{"id":"r{i}","paths":["a"],"minimum":"standard"}}' for i in range(65)
    )
    limits_dir = tmp_path / "limits"
    limits_dir.mkdir()
    limits_repo = _repo(limits_dir)
    limits_base = _commit(
        limits_repo,
        {"REVIEW.md": _policy('{"rules":[' + too_many_rules + "]}"), "a": "a"},
    )
    limits_head = _commit(limits_repo, {"a": "b"})
    with pytest.raises(ActionError, match="rules or globs"):
        resolve_policy(limits_repo, limits_base, limits_head)
    paths = ",".join(f'"a{i}"' for i in range(17))
    with pytest.raises(ActionError, match="1 through 16"):
        glob_policy = '{"rules":[{"id":"x","paths":[' + paths + '],"minimum":"deep"}]}'
        parse_policy_file(
            "REVIEW.md",
            _policy(glob_policy).encode(),
        )

    repo = _repo(tmp_path)
    base = _commit(
        repo,
        {
            "REVIEW.md": _policy('{"minimum":"deep"}'),
            "src/REVIEW.md": _policy('{"minimum":"standard"}'),
            "src/a": "a",
        },
    )
    head = _commit(repo, {"src/a": "b"})
    assert resolve_policy(repo, base, head).minimum == "deep"
    deep_path = "/".join(["d"] * 21)
    with pytest.raises(ActionError, match="ancestor depth"):
        resolve_policy(repo, base, head, (deep_path,))


@pytest.mark.parametrize("mode, name", [("120000", "link"), ("160000", "sub")])
def test_applicable_symlinks_and_submodules_are_rejected(
    tmp_path: Path, mode: str, name: str
) -> None:
    repo = _repo(tmp_path)
    _commit(repo, {"a": "a"})
    object_id = (
        _git(repo, "rev-parse", "HEAD")
        if mode == "160000"
        else _git(repo, "hash-object", "-w", "--stdin", input="target")
    )
    _git(repo, "update-index", "--add", "--cacheinfo", f"{mode},{object_id},{name}")
    _git(repo, "commit", "-m", "special entry")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "rm", "--cached", name)
    (repo / name).write_text("ordinary now", encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-m", "change special entry")
    with pytest.raises(ActionError, match="traverses"):
        resolve_policy(repo, base, _git(repo, "rev-parse", "HEAD"))


def test_git_failure_includes_redacted_actionable_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    _commit(repo, {"a": "a"})
    secret = "ghp_supersecrettokenvalue123456"
    monkeypatch.setenv("GITHUB_TOKEN", secret)
    _patch_git_popen(monkeypatch, tmp_path, "fail_stderr")
    with pytest.raises(ActionError, match=r"git policy discovery cat-file -t failed") as exc:
        bounded_git(repo, ["cat-file", "-t", "0" * 40], 10)
    message = str(exc.value)
    assert secret not in message
    assert "[redacted]" in message
    assert "missing object" in message


def _blob(repo: Path, content: str) -> str:
    return (
        subprocess.run(
            ["git", "-C", str(repo), "hash-object", "-w", "--stdin"],
            input=content.encode("utf-8"),
            capture_output=True,
            check=True,
        )
        .stdout.decode()
        .strip()
    )


def _mktree(repo: Path, *entries: str) -> str:
    return _git(
        repo, "mktree", "-z", input=b"\0".join(entry.encode("utf-8") for entry in entries) + b"\0"
    )


def _mktree_bytes(repo: Path, content: bytes) -> str:
    return (
        subprocess.run(
            ["git", "-C", str(repo), "mktree", "-z"],
            check=True,
            input=content,
            capture_output=True,
        )
        .stdout.decode()
        .strip()
    )


def test_discovery_rejects_invalid_carried_and_changed_paths(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = _commit(repo, {"a": "a"})
    head = _commit(repo, {"a": "b"})
    with pytest.raises(ActionError, match="carried.*backticks"):
        resolve_policy(repo, base, head, ("src/`evil`.py",))
    long_path = "a/" * 300 + "file.py"
    with pytest.raises(ActionError, match="carried path exceeds"):
        resolve_policy(repo, base, head, (long_path,))
    with pytest.raises(ActionError, match="not canonical"):
        resolve_policy(repo, base, head, ("a//b.py",))
    with pytest.raises(ActionError, match="not canonical"):
        _discovery_path(" src/a.py", "changed")


def test_whitespace_review_md_does_not_alias_canonical(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    parent = _commit(repo, {".gitkeep": ""})
    canonical = _blob(repo, "canonical root\n")
    whitespace = _blob(repo, "evil substitute\n")
    a_blob = _blob(repo, "a\n")
    src_tree = _mktree(repo, f"100644 blob {a_blob}\ta.py")
    root_tree = _mktree(
        repo,
        f"100644 blob {canonical}\tREVIEW.md",
        f"100644 blob {whitespace}\t REVIEW.md",
        f"040000 tree {src_tree}\tsrc",
    )
    base = _git(repo, "commit-tree", root_tree, "-p", parent, "-m", "base")
    b_blob = _blob(repo, "b\n")
    src_tree_head = _mktree(repo, f"100644 blob {b_blob}\ta.py")
    head_tree = _mktree(
        repo,
        f"100644 blob {canonical}\tREVIEW.md",
        f"100644 blob {whitespace}\t REVIEW.md",
        f"040000 tree {src_tree_head}\tsrc",
    )
    head = _git(repo, "commit-tree", head_tree, "-p", base, "-m", "head")
    found = resolve_policy(repo, base, head)
    assert [item.path for item in found.files] == ["REVIEW.md"]
    assert found.files[0].content == "canonical root\n"


def test_discover_entries_visits_ancestors_in_order(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    parent = _commit(repo, {".gitkeep": ""})
    root_review = _blob(repo, "root\n")
    nested_review = _blob(repo, "nested\n")
    deep_review = _blob(repo, "deep\n")
    leaf_blob = _blob(repo, "x\n")
    deep_tree = _mktree(repo, f"100644 blob {deep_review}\tREVIEW.md")
    nested_tree = _mktree(
        repo,
        f"100644 blob {nested_review}\tREVIEW.md",
        f"040000 tree {deep_tree}\tsrc-nested-deep",
    )
    src_tree = _mktree(
        repo,
        f"100644 blob {nested_review}\tREVIEW.md",
        f"100644 blob {leaf_blob}\ta.py",
        f"040000 tree {nested_tree}\tsrc-nested",
    )
    root_tree = _mktree(
        repo,
        f"100644 blob {root_review}\tREVIEW.md",
        f"040000 tree {src_tree}\tsrc",
    )
    base = _git(repo, "commit-tree", root_tree, "-p", parent, "-m", "base")
    directories = {"src/src-nested/src-nested-deep", "src", "src/src-nested", ""}
    entries = _discover_entries(repo, base, directories, timeout=120)
    assert set(entries) >= {
        "REVIEW.md",
        "src/REVIEW.md",
        "src/src-nested/REVIEW.md",
        "src/src-nested/src-nested-deep/REVIEW.md",
    }


def test_unrelated_subtree_is_not_recursively_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    parent = _commit(repo, {".gitkeep": ""})
    blobs = {
        "root": _blob(repo, "root policy\n"),
        "src_review": _blob(repo, "src policy\n"),
        "a": _blob(repo, "a\n"),
        "b": _blob(repo, "b\n"),
        "x": _blob(repo, "x\n"),
        "n": _blob(repo, "n\n"),
    }
    noise_entries = [f"100644 blob {blobs['n']}\tnoise{i}.txt" for i in range(40)]
    noise_entries.append(f"100644 blob {blobs['x']}\tbad:name")
    noise_entries.append(f"100644 blob {blobs['x']}\t`weird`.md")
    unrelated_tree = _mktree(repo, *noise_entries)
    src_tree = _mktree(
        repo,
        f"100644 blob {blobs['src_review']}\tREVIEW.md",
        f"100644 blob {blobs['a']}\ta.py",
    )
    root_tree = _mktree(
        repo,
        f"100644 blob {blobs['root']}\tREVIEW.md",
        f"040000 tree {src_tree}\tsrc",
        f"040000 tree {unrelated_tree}\tunrelated",
    )
    base = _git(repo, "commit-tree", root_tree, "-p", parent, "-m", "base")
    src_tree_head = _mktree(
        repo,
        f"100644 blob {blobs['src_review']}\tREVIEW.md",
        f"100644 blob {blobs['b']}\ta.py",
    )
    head_tree = _mktree(
        repo,
        f"100644 blob {blobs['root']}\tREVIEW.md",
        f"040000 tree {src_tree_head}\tsrc",
        f"040000 tree {unrelated_tree}\tunrelated",
    )
    head = _git(repo, "commit-tree", head_tree, "-p", base, "-m", "head")

    listed: list[str] = []
    real_git = bounded_git

    def tracking_git(
        repo_path: Path, args: list[str], timeout: int, limit: int = _GIT_OUTPUT
    ) -> bytes:
        if len(args) >= 2 and args[0] == "ls-tree" and args[1] == "--full-tree":
            listed.append(args[-1])
        return real_git(repo_path, args, timeout, limit)

    monkeypatch.setattr("or_pr_review.review_policy._git", tracking_git)
    found = resolve_policy(repo, base, head)
    assert [item.path for item in found.files] == ["REVIEW.md", "src/REVIEW.md"]
    assert found.files[1].scope_paths == ("src/a.py",)
    assert len(listed) == 2


def test_undecodable_irrelevant_tree_names_are_skipped(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    parent = _commit(repo, {".gitkeep": ""})
    root_review = _blob(repo, "root policy\n")
    src_review = _blob(repo, "src policy\n")
    src_py = _blob(repo, "a\n")
    changed_py = _blob(repo, "b\n")
    noise_blob = _blob(repo, "n\n")
    bad_name = b"\xffnoise.txt"
    src_tree = _mktree_bytes(
        repo,
        b"\0".join(
            [
                f"100644 blob {src_review}\tREVIEW.md".encode(),
                f"100644 blob {src_py}\ta.py".encode(),
                f"100644 blob {noise_blob}\t".encode() + bad_name,
            ]
        )
        + b"\0",
    )
    unrelated_tree = _mktree(repo, f"100644 blob {noise_blob}\tnoise.txt")
    root_tree = _mktree(
        repo,
        f"100644 blob {root_review}\tREVIEW.md",
        f"040000 tree {src_tree}\tsrc",
        f"040000 tree {unrelated_tree}\tunrelated",
    )
    base = _git(repo, "commit-tree", root_tree, "-p", parent, "-m", "base")
    src_tree_head = _mktree_bytes(
        repo,
        b"\0".join(
            [
                f"100644 blob {src_review}\tREVIEW.md".encode(),
                f"100644 blob {changed_py}\ta.py".encode(),
                f"100644 blob {noise_blob}\t".encode() + bad_name,
            ]
        )
        + b"\0",
    )
    head_tree = _mktree(
        repo,
        f"100644 blob {root_review}\tREVIEW.md",
        f"040000 tree {src_tree_head}\tsrc",
        f"040000 tree {unrelated_tree}\tunrelated",
    )
    head = _git(repo, "commit-tree", head_tree, "-p", base, "-m", "head")
    found = resolve_policy(repo, base, head)
    assert [item.path for item in found.files] == ["REVIEW.md", "src/REVIEW.md"]


@pytest.mark.parametrize("policy_name", ["REVIEW.md", "review.md"])
def test_overlong_applicable_review_name_is_rejected(tmp_path: Path, policy_name: str) -> None:
    repo = _repo(tmp_path)
    parent = _commit(repo, {".gitkeep": ""})
    components = [f"segment{i:02d}" + "x" * 16 for i in range(19)]
    directory = "/".join(components)
    assert len(directory) <= 500 < len(directory) + 1 + len(policy_name)
    policy_blob = _blob(repo, "policy\n")
    old_blob = _blob(repo, "old\n")
    new_blob = _blob(repo, "new\n")
    tree = _mktree(
        repo,
        f"100644 blob {policy_blob}\t{policy_name}",
        f"100644 blob {old_blob}\ta.py",
    )
    for component in reversed(components):
        tree = _mktree(repo, f"040000 tree {tree}\t{component}")
    base = _git(repo, "commit-tree", tree, "-p", parent, "-m", "base")
    head_tree = _mktree(
        repo,
        f"100644 blob {policy_blob}\t{policy_name}",
        f"100644 blob {new_blob}\ta.py",
    )
    for component in reversed(components):
        head_tree = _mktree(repo, f"040000 tree {head_tree}\t{component}")
    head = _git(repo, "commit-tree", head_tree, "-p", base, "-m", "head")
    with pytest.raises(ActionError, match="base tree path exceeds 500"):
        resolve_policy(repo, base, head)


def test_unrelated_odd_subtree_does_not_break_discovery(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    parent = _commit(repo, {".gitkeep": ""})
    blobs = {
        "root": _blob(repo, "root policy\n"),
        "src_review": _blob(repo, "src policy\n"),
        "a": _blob(repo, "a\n"),
        "b": _blob(repo, "b\n"),
        "x": _blob(repo, "x\n"),
        "n": _blob(repo, "n\n"),
    }
    noise_entries = [f"100644 blob {blobs['n']}\tnoise{i}.txt" for i in range(40)]
    noise_entries.append(f"100644 blob {blobs['x']}\tbad:name")
    noise_entries.append(f"100644 blob {blobs['x']}\t`weird`.md")
    unrelated_tree = _mktree(repo, *noise_entries)
    src_tree = _mktree(
        repo,
        f"100644 blob {blobs['src_review']}\tREVIEW.md",
        f"100644 blob {blobs['a']}\ta.py",
    )
    root_tree = _mktree(
        repo,
        f"100644 blob {blobs['root']}\tREVIEW.md",
        f"040000 tree {src_tree}\tsrc",
        f"040000 tree {unrelated_tree}\tunrelated",
    )
    base = _git(repo, "commit-tree", root_tree, "-p", parent, "-m", "base")
    src_tree_head = _mktree(
        repo,
        f"100644 blob {blobs['src_review']}\tREVIEW.md",
        f"100644 blob {blobs['b']}\ta.py",
    )
    head_tree = _mktree(
        repo,
        f"100644 blob {blobs['root']}\tREVIEW.md",
        f"040000 tree {src_tree_head}\tsrc",
        f"040000 tree {unrelated_tree}\tunrelated",
    )
    head = _git(repo, "commit-tree", head_tree, "-p", base, "-m", "head")
    found = resolve_policy(repo, base, head)
    assert [item.path for item in found.files] == ["REVIEW.md", "src/REVIEW.md"]
    assert found.files[1].scope_paths == ("src/a.py",)


def test_unrelated_case_variant_is_ignored(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = _commit(
        repo,
        {
            "REVIEW.md": "root\n",
            "src/a.py": "a\n",
            "unrelated/reView.md": "wrong case but irrelevant\n",
        },
    )
    head = _commit(repo, {"src/a.py": "b\n"})
    assert resolve_policy(repo, base, head).files[0].path == "REVIEW.md"


def test_root_and_nested_policies_discovered_with_bounded_walk(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = _commit(
        repo,
        {
            "REVIEW.md": "root\n",
            "src/REVIEW.md": "nested\n",
            "src/a.py": "a\n",
            "other/b.py": "b\n",
        },
    )
    head = _commit(repo, {"src/a.py": "changed\n"})
    found = resolve_policy(repo, base, head)
    assert [item.path for item in found.files] == ["REVIEW.md", "src/REVIEW.md"]
    assert found.files[0].scope_paths == ("src/a.py",)
    assert found.files[1].scope_paths == ("src/a.py",)
