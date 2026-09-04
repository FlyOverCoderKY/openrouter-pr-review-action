from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from or_pr_review.workspace import (
    MAX_GREP_MATCHES,
    MAX_LIST_ENTRIES,
    _safe_member,
    dispatch_tool,
    is_blocked_path,
    tool_grep,
    tool_read_file,
)


def test_read_and_grep_and_list(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    assert "def add" in tool_read_file(tmp_path, "src/app.py")
    assert "src/app.py:1" in tool_grep(tmp_path, r"def add")
    listing = dispatch_tool(tmp_path, "list_dir", {"path": "src"})
    assert "app.py" in listing


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    result = dispatch_tool(tmp_path, "read_file", {"path": "../secret"})
    assert "escapes" in result or "error" in result


def test_secret_like_paths_blocked(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=sk-or-v1-secret\n", encoding="utf-8")
    assert is_blocked_path(env_file)
    assert "refusing" in tool_read_file(tmp_path, ".env")


@pytest.mark.parametrize(
    "relative",
    [
        "nested/credentials.json",
        "NESTED/Secrets/review.txt",
        "nested/private-key.PEM",
        "nested/service-account.JSON",
    ],
)
def test_nested_and_case_variant_credential_paths_are_blocked(
    tmp_path: Path, relative: str
) -> None:
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("TOP-SECRET\n", encoding="utf-8")
    assert is_blocked_path(target)
    assert "refusing" in tool_read_file(tmp_path, relative)


def test_dotenv_contents_refused(tmp_path: Path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("FOO=1\nBAR=2\nBAZ=3\n", encoding="utf-8")
    assert "refusing" in tool_read_file(tmp_path, "notes.txt")


def test_unknown_tool_rejected(tmp_path: Path) -> None:
    assert "disallowed" in dispatch_tool(tmp_path, "shell", {"cmd": "ls"})


def test_ranged_read_returns_window(tmp_path: Path) -> None:
    lines = "".join(f"line {n}\n" for n in range(1, 11))
    (tmp_path / "big.txt").write_text(lines, encoding="utf-8")
    result = tool_read_file(tmp_path, "big.txt", start_line=3, max_lines=2)
    assert result.startswith("[lines 3-4 of 10]")
    assert "line 3" in result and "line 4" in result
    assert "line 5" not in result
    assert "start_line=5" in result  # continuation hint


def test_unranged_large_file_truncates_with_continuation_hint(tmp_path: Path) -> None:
    from or_pr_review.workspace import MAX_READ_BYTES

    (tmp_path / "huge.txt").write_text("padding words here\n" * 5000, encoding="utf-8")
    result = tool_read_file(tmp_path, "huge.txt")
    assert f"[truncated after {MAX_READ_BYTES} bytes" in result
    assert "start_line=" in result
    assert len(result.encode("utf-8")) < MAX_READ_BYTES + 300


def test_ranged_read_rejects_bad_arguments(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
    assert "error: start_line" in dispatch_tool(
        tmp_path, "read_file", {"path": "a.txt", "start_line": "abc"}
    )
    assert "error: max_lines" in dispatch_tool(
        tmp_path, "read_file", {"path": "a.txt", "max_lines": 1.5}
    )
    assert "past the end" in tool_read_file(tmp_path, "a.txt", start_line=99)


def test_ranged_read_accepts_digit_strings(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    result = dispatch_tool(tmp_path, "read_file", {"path": "a.txt", "start_line": "2"})
    assert result.startswith("[lines 2-3 of 3]")


def test_ranged_read_reports_only_delivered_lines_when_byte_truncated(
    tmp_path: Path,
) -> None:
    from or_pr_review.workspace import MAX_READ_BYTES

    line = "x" * 199 + "\n"  # 200 bytes per line
    total_lines = 500
    (tmp_path / "big.txt").write_text(line * total_lines, encoding="utf-8", newline="\n")
    result = tool_read_file(tmp_path, "big.txt", start_line=1, max_lines=total_lines)
    delivered = MAX_READ_BYTES // 200
    assert result.startswith(f"[lines 1-{delivered} of {total_lines}]")
    assert f"start_line={delivered + 1}" in result
    assert f"1-{total_lines} of" not in result  # never claim undelivered lines


def test_bad_int_arguments_are_tool_errors_not_crashes(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
    # These pass isdigit()-style pre-checks but make int() raise; they must
    # come back as tool error observations, never exceptions.
    for bad in ("--3", "①", "1.5", "٣٣x"):
        result = dispatch_tool(tmp_path, "read_file", {"path": "a.txt", "start_line": bad})
        assert result.startswith("error:"), bad


def test_list_dir_caps_entries(tmp_path: Path) -> None:
    for index in range(MAX_LIST_ENTRIES + 5):
        (tmp_path / f"file-{index:03}.txt").write_text("x\n", encoding="utf-8")
    result = dispatch_tool(tmp_path, "list_dir", {})
    assert f"[{5} more entries omitted]" in result
    assert f"file-{MAX_LIST_ENTRIES + 4:03}.txt" not in result


def test_grep_caps_matches(tmp_path: Path) -> None:
    (tmp_path / "many.txt").write_text("needle\n" * (MAX_GREP_MATCHES + 10), encoding="utf-8")
    result = tool_grep(tmp_path, "needle")
    assert "[grep match cap reached]" in result
    assert result.count(":needle") == MAX_GREP_MATCHES


def test_symlink_files_are_not_exposed(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("needle\n", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    assert "link.txt" not in dispatch_tool(tmp_path, "list_dir", {})
    assert "symlink" in dispatch_tool(tmp_path, "read_file", {"path": "link.txt"})
    assert "link.txt" not in tool_grep(tmp_path, "needle")


@pytest.mark.parametrize(
    "name",
    ["../escape.txt", "/absolute.txt", "safe/../../escape.txt", r"safe\..\escape.txt"],
)
def test_tar_traversal_members_are_rejected(name: str) -> None:
    member = tarfile.TarInfo(name)
    member.size = 0
    assert not _safe_member(member)


def test_tar_links_are_rejected() -> None:
    member = tarfile.TarInfo("safe.txt")
    member.type = tarfile.SYMTYPE
    member.linkname = "../outside.txt"
    assert not _safe_member(member)


def test_tar_regular_member_is_allowed() -> None:
    member = tarfile.TarInfo("safe/path.txt")
    member.size = 10
    assert _safe_member(member)


def test_tool_worker_reports_invalid_input_on_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import or_pr_review.tool_worker as worker

    monkeypatch.setattr(worker.sys, "argv", ["tool_worker"])
    assert worker.main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage:" in captured.err

    monkeypatch.setattr(worker.sys, "argv", ["tool_worker", str(Path(".")), "read_file"])
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO("not-json"))
    assert worker.main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid JSON" in captured.err


def test_tool_worker_returns_json_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import or_pr_review.tool_worker as worker

    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    monkeypatch.setattr(worker.sys, "argv", ["tool_worker", str(tmp_path), "read_file"])
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO(json.dumps({"path": "a.txt"})))
    assert worker.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["result"].replace("\r\n", "\n") == "hello\n"
    assert captured.err == ""


def test_unexpected_tool_exception_becomes_error_observation(tmp_path: Path, monkeypatch) -> None:
    import or_pr_review.workspace as ws

    def boom(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(ws, "tool_read_file", boom)
    result = dispatch_tool(tmp_path, "read_file", {"path": "a.txt"})
    assert result.startswith("error: tool 'read_file' failed: RuntimeError")


def test_materialize_commit_writes_full_manifest_despite_skips(tmp_path):
    import subprocess

    from or_pr_review.workspace import materialize_commit, tracked_paths

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            [
                "git",
                "-c",
                "user.email=t@example.invalid",
                "-c",
                "user.name=T",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    git("init", "-q", "-b", "main")
    (repo / "small.py").write_text("print('hi')\n", encoding="utf-8")
    (repo / "big.bin").write_bytes(b"\0" * (1_100_000))  # over the 1MB cap
    git("add", "-A")
    git("commit", "-q", "-m", "c")
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()

    dest = tmp_path / "inert-checkout"
    materialize_commit(repo, sha, dest)
    # The oversized member is skipped on disk but present in the manifest, so
    # the anchor gate can tell a snapshot hole from a ghost path.
    assert not (dest / "big.bin").exists()
    assert (dest / "small.py").is_file()
    manifest = tracked_paths(dest)
    assert manifest == {"small.py", "big.bin"}


def test_materialize_commit_allows_oversized_stubbed_files(tmp_path):
    import subprocess

    from or_pr_review.workspace import materialize_commit, tool_grep, tool_read_file

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            [
                "git",
                "-c",
                "user.email=t@example.invalid",
                "-c",
                "user.name=T",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    git("init", "-q", "-b", "main")
    # Over the 1MB default cap; the class of file diff-budget triage stubs.
    (repo / "snapshot.json").write_text(
        '{"needle": "rgorg-plant"}' + "x" * 1_100_000, encoding="utf-8"
    )
    (repo / "huge.bin").write_bytes(b"\0" * 1_100_000)
    git("add", "-A")
    git("commit", "-q", "-m", "c")
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()

    dest = tmp_path / "inert-checkout"
    materialize_commit(repo, sha, dest, oversized_ok=frozenset({"snapshot.json"}))
    # The stubbed file honors its tool-readability contract...
    assert (dest / "snapshot.json").is_file()
    assert tool_read_file(dest, "snapshot.json").startswith('{"needle"')
    assert "snapshot.json:1:" in tool_grep(dest, "rgorg-plant")
    # ...while unlisted oversized files keep the normal cap.
    assert not (dest / "huge.bin").exists()
