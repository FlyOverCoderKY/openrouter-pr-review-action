from __future__ import annotations

from pathlib import Path

from or_pr_review.workspace import dispatch_tool, is_blocked_path, tool_grep, tool_read_file


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
