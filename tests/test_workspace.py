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
