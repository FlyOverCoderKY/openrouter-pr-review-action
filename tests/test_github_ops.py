from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from or_pr_review.errors import ActionError
from or_pr_review.github_ops import GitHub


def test_run_gh_maps_timeout_to_action_error(monkeypatch) -> None:
    from or_pr_review import github_ops as github_ops_mod

    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(github_ops_mod, "_bounded_command", run)

    with pytest.raises(ActionError, match=r"GitHub CLI timed out after 17s"):
        github_ops_mod._run_gh(["gh", "api", "user"], env={}, timeout=17)


def test_run_gh_maps_os_error_to_action_error(monkeypatch) -> None:
    from or_pr_review import github_ops as github_ops_mod

    def run(*args, **kwargs):
        raise FileNotFoundError("gh executable not found")

    monkeypatch.setattr(github_ops_mod, "_bounded_command", run)

    with pytest.raises(ActionError, match=r"failed to run GitHub CLI: gh executable not found"):
        github_ops_mod._run_gh(["gh", "api", "user"], env={}, timeout=17)


def test_run_gh_redacts_failed_command_stderr(monkeypatch) -> None:
    from or_pr_review import github_ops as github_ops_mod

    secret = "github-token-that-must-not-leak"
    monkeypatch.setenv("GH_TOKEN", secret)
    completed = (1, b"", f"request failed with token={secret}".encode())
    monkeypatch.setattr(github_ops_mod, "_bounded_command", lambda *args, **kwargs: completed)

    with pytest.raises(ActionError) as raised:
        github_ops_mod._run_gh(["gh", "api", "user"], env={}, timeout=17)

    assert secret not in str(raised.value)
    assert "token=[redacted]" in str(raised.value)


def test_github_env_excludes_openrouter_key_and_preserves_runner_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "paid-provider-secret")
    monkeypatch.setenv("RUNNER_CONTEXT", "preserved")

    env = GitHub(token="github-secret", repository="o/r")._env()

    assert "OPENROUTER_API_KEY" not in env
    assert env["RUNNER_CONTEXT"] == "preserved"
    assert env["GH_TOKEN"] == "github-secret"
    assert env["GITHUB_TOKEN"] == "github-secret"
    assert env["GH_PROMPT_DISABLED"] == "1"
    assert env is not os.environ


@pytest.mark.parametrize("sha_length", [7, 40])
def test_compare_diff_validates_fast_forward_and_fetches_raw(sha_length) -> None:
    calls: list[list[str]] = []

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        calls.append(cmd)
        if "-H" in cmd:
            return "raw-diff-text"
        return json.dumps({"status": "ahead", "behind_by": 0})

    gh = GitHub(token="t", repository="o/r", runner=runner)
    assert gh.compare_diff("a" * sha_length, "b" * sha_length) == "raw-diff-text"
    assert len(calls) == 2
    assert "Accept: application/vnd.github.diff" in calls[1]


def test_compare_diff_rejects_non_fast_forward_with_distinct_error() -> None:
    from or_pr_review.errors import DivergedRangeError

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        return json.dumps({"status": "diverged", "behind_by": 3})

    gh = GitHub(token="t", repository="o/r", runner=runner)
    # The distinct type matters: only a genuine divergence may reset the
    # review loop; transport failures raise plain ActionError.
    with pytest.raises(DivergedRangeError, match="fast-forward"):
        gh.compare_diff("a" * 40, "b" * 40)


@pytest.mark.parametrize("sha_length", [7, 40])
def test_commit_diff_uses_raw_media_type(sha_length) -> None:
    calls: list[list[str]] = []

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        calls.append(cmd)
        return "raw-commit-diff"

    gh = GitHub(token="t", repository="o/r", runner=runner)
    assert gh.commit_diff("c" * sha_length) == "raw-commit-diff"
    assert "Accept: application/vnd.github.diff" in calls[0]


@pytest.mark.parametrize(
    "too_large_message",
    [
        "HTTP 406: Sorry, the diff exceeded the maximum number of lines (20000)",
        "HTTP 406: PullRequest.diff too_large",
    ],
)
def test_pr_diff_falls_back_to_local_git_on_github_line_limit(
    monkeypatch, too_large_message: str
) -> None:
    base = "a" * 40
    head = "b" * 40
    gh_calls: list[list[str]] = []
    git_calls: list[tuple[list[str], int, str]] = []

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        gh_calls.append(cmd)
        assert cmd[1:4] == ["api", "-H", "Accept: application/vnd.github.diff"]
        assert cmd[-1] == f"repos/o/r/compare/{base}...{head}"
        raise ActionError(too_large_message)

    def git_runner(cmd: list[str], *, timeout: int, cwd: str) -> str:
        git_calls.append((cmd, timeout, cwd))
        return "complete-local-diff"

    gh = GitHub(
        token="t",
        repository="o/r",
        timeout=45,
        runner=runner,
        git_runner=git_runner,
        source_workspace="/checkout",
    )
    assert gh.pr_diff(7, base_sha=base, head_sha=head) == "complete-local-diff"
    assert len(gh_calls) == 1
    assert git_calls == [
        (
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
            45,
            "/checkout",
        )
    ]


def test_pr_diff_does_not_hide_other_github_failures() -> None:
    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        raise ActionError("HTTP 403: resource not accessible")

    def git_runner(cmd: list[str], *, timeout: int, cwd: str) -> str:
        raise AssertionError("local git must not run")

    gh = GitHub(token="t", repository="o/r", runner=runner, git_runner=git_runner)
    with pytest.raises(ActionError, match="HTTP 403"):
        gh.pr_diff(7, base_sha="a" * 40, head_sha="b" * 40)


def test_list_bot_review_bodies_filters_author() -> None:
    reviews = [
        [
            {"user": {"login": "github-actions[bot]"}, "body": "mine"},
            {"user": {"login": "someone"}, "body": "not mine"},
            {"user": {"login": "github-actions[bot]"}, "body": None},
        ]
    ]

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        assert "--paginate" in cmd and "--slurp" in cmd
        return json.dumps(reviews)

    gh = GitHub(token="t", repository="o/r", runner=runner)
    assert gh.list_bot_review_bodies(1, "github-actions[bot]") == ["mine"]


def test_newest_ledger_after_reply_filled_first_page_is_recovered():
    from or_pr_review.loop import Ledger, encode_ledger, latest_ledger

    old = encode_ledger(Ledger(1, (), "a" * 40, "a" * 12), repo="o/r", pr_number=1)
    new = encode_ledger(Ledger(3, (), "b" * 40, "a" * 12), repo="o/r", pr_number=1)
    pages = [
        [{"user": {"login": "github-actions[bot]"}, "body": old}]
        + [{"user": {"login": "developer"}, "body": "fixed"}] * 29,
        [{"user": {"login": "github-actions[bot]"}, "body": new}],
    ]

    def runner(cmd, **kwargs):
        assert "--paginate" in cmd and "--slurp" in cmd
        return json.dumps(pages)

    github = GitHub(token="t", repository="o/r", runner=runner)
    ledger = latest_ledger(
        github.list_bot_review_bodies(1, "github-actions[bot]"), repo="o/r", pr_number=1
    )
    assert ledger.reviewed_sha == "b" * 40
    assert ledger.round_number == 3


def test_create_review_falls_back_without_comments() -> None:
    calls: list[dict] = []

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        calls.append(json.loads(stdin or "{}"))
        if len(calls) == 1:
            raise ActionError("422 line is not part of the diff")
        return json.dumps({"html_url": "https://example.test/review"})

    gh = GitHub(token="t", repository="o/r", runner=runner)
    result = gh.create_review(
        1,
        "body",
        "c" * 40,
        comments=[{"path": "a.py", "line": 1, "side": "RIGHT", "body": "x"}],
    )
    assert result["html_url"] == "https://example.test/review"
    assert "comments" in calls[0]
    assert "comments" not in calls[1]


def test_create_review_does_not_retry_after_transport_failure() -> None:
    calls: list[dict] = []

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        calls.append(json.loads(stdin or "{}"))
        raise ActionError("GitHub CLI timed out after 120s")

    gh = GitHub(token="t", repository="o/r", runner=runner)
    with pytest.raises(ActionError, match="timed out"):
        gh.create_review(
            1,
            "body",
            "c" * 40,
            comments=[{"path": "a.py", "line": 1, "side": "RIGHT", "body": "x"}],
        )
    assert len(calls) == 1


def test_upsert_status_comment_uses_paginated_issue_comments() -> None:
    from or_pr_review.github_ops import STATUS_MARKER, upsert_status_comment

    calls: list[list[str]] = []

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        calls.append(cmd)
        if "--paginate" in cmd:
            return json.dumps(
                [
                    [
                        {
                            "id": 101,
                            "body": f"{STATUS_MARKER}\nold",
                            "user": {"login": "github-actions[bot]"},
                        }
                    ]
                ]
            )
        return json.dumps({"html_url": "https://example.test/status"})

    gh = GitHub(token="t", repository="o/r", runner=runner)
    assert upsert_status_comment(gh, pr_number=1, body="new") == "https://example.test/status"
    assert "--paginate" in calls[0]
    assert "PATCH" in calls[1]


def test_upsert_status_comment_creates_when_marker_is_absent() -> None:
    from or_pr_review.github_ops import STATUS_MARKER, upsert_status_comment

    requests: list[tuple[list[str], str | None]] = []

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        requests.append((cmd, stdin))
        if "--paginate" in cmd:
            return json.dumps([[{"id": 100, "body": "an unrelated comment"}]])
        return json.dumps({"html_url": "https://example.test/new-status"})

    gh = GitHub(token="t", repository="o/r", runner=runner)

    assert upsert_status_comment(gh, pr_number=7, body="running") == (
        "https://example.test/new-status"
    )
    assert "--paginate" in requests[0][0]
    assert requests[1][0][1:3] == ["api", "repos/o/r/issues/7/comments"]
    assert requests[1][0][-2:] == ["--input", "-"]
    assert json.loads(requests[1][1] or "{}") == {"body": f"{STATUS_MARKER}\nrunning\n"}


def test_list_finding_replies_pairs_marker_threads_by_generation() -> None:
    generation = "1234567890ab"
    other_generation = "feedfeedfeed"
    comments = [
        [
            {
                "id": 10,
                "body": f"<!-- or-finding:{generation}:r1-1 -->\n**T**",
                "user": {"login": "bot"},
            },
            {"id": 11, "in_reply_to_id": 10, "body": "fixed in abc123", "user": {"login": "dev"}},
            {
                "id": 20,
                "body": f"<!-- or-finding:{other_generation}:r1-1 -->\n**Old**",
                "user": {"login": "bot"},
            },
            {
                "id": 21,
                "in_reply_to_id": 20,
                "body": "false positive because X",
                "user": {"login": "dev"},
            },
            {"id": 12, "in_reply_to_id": 99, "body": "unrelated", "user": {"login": "dev"}},
        ]
    ]

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        return json.dumps(comments)

    gh = GitHub(token="t", repository="o/r", runner=runner)
    # Only the current generation's thread pairs; the pre-reset thread with a
    # reused finding id is ignored, and an empty generation harvests nothing.
    assert gh.list_finding_replies(
        1, generation=generation, bot_login="bot", carried_ids={"r1-1"}
    ) == [("r1-1", "dev", "fixed in abc123")]
    assert gh.list_finding_replies(1, generation="", bot_login="bot", carried_ids={"r1-1"}) == []


@pytest.mark.parametrize(
    "root_changes",
    [
        {"user": {"login": "developer"}},
        {"user": {}},
        {"in_reply_to_id": 99},
        {"body": "<!-- or-finding:abcdef123456:r1-2 -->\nold finding"},
        {"body": "prose\n<!-- or-finding:abcdef123456:r1-1 -->"},
    ],
)
def test_finding_replies_require_bot_root_and_carried_id(root_changes):
    root = {
        "id": 10,
        "body": "<!-- or-finding:abcdef123456:r1-1 -->\nfinding",
        "user": {"login": "custom-bot[bot]"},
        **root_changes,
    }
    comments = [root, {"id": 11, "in_reply_to_id": 10, "body": "reply"}]
    gh = GitHub(token="t", repository="o/r", runner=lambda *a, **kw: json.dumps(comments))
    assert (
        gh.list_finding_replies(
            1, generation="abcdef123456", bot_login="custom-bot[bot]", carried_ids={"r1-1"}
        )
        == []
    )


@pytest.mark.parametrize("include_bot", [False, True])
def test_status_upsert_selects_only_configured_bot(include_bot):
    from or_pr_review.github_ops import STATUS_MARKER, upsert_status_comment

    comments = [
        {"id": 10, "body": STATUS_MARKER, "user": {"login": "developer"}},
        {"id": 11, "body": STATUS_MARKER, "user": {"login": "github-actions[bot]"}},
        {"id": 12, "body": STATUS_MARKER},
    ]
    if include_bot:
        comments.append({"id": 20, "body": STATUS_MARKER, "user": {"login": "custom-bot"}})
    writes = []

    def runner(cmd, **kwargs):
        if "--paginate" in cmd:
            return json.dumps(comments)
        writes.append(cmd)
        return json.dumps({"html_url": "https://example.test/status"})

    gh = GitHub(token="t", repository="o/r", runner=runner)
    upsert_status_comment(gh, pr_number=1, body="running", bot_login="custom-bot")
    assert len(writes) == 1
    if include_bot:
        assert "PATCH" in writes[0]
        assert "repos/o/r/issues/comments/20" in writes[0]
    else:
        assert "PATCH" not in writes[0]
        assert "repos/o/r/issues/1/comments" in writes[0]


def test_recent_issue_comments_exclude_bot_review_bodies() -> None:
    comments = [
        [
            {
                "id": 1,
                "body": "## OpenRouter pull-request review — continued\n...",
                "user": {"login": "b"},
            },
            {"id": 2, "body": "## OpenRouter review incomplete\n...", "user": {"login": "b"}},
            {"id": 3, "body": "I pushed a fix for r1-1", "user": {"login": "dev"}},
        ]
    ]

    def runner(cmd: list[str], *, env: dict, timeout: int, stdin: str | None = None) -> str:
        return json.dumps(comments)

    gh = GitHub(token="t", repository="o/r", runner=runner)
    assert gh.list_recent_issue_comments(1) == [("dev", "I pushed a fix for r1-1")]


def test_full_pr_diff_requests_only_the_pinned_range() -> None:
    calls = []
    base, head = "a" * 40, "b" * 40

    def runner(cmd, **kwargs):
        calls.append(cmd)
        assert cmd[-1] == f"repos/o/r/compare/{base}...{head}"
        return "pinned-diff"

    gh = GitHub(token="t", repository="o/r", runner=runner)
    assert gh.pr_diff(7, base_sha=base.upper(), head_sha=head.upper()) == "pinned-diff"
    assert len(calls) == 1
    assert "Accept: application/vnd.github.diff" in calls[0]


@pytest.mark.parametrize("base,head", [(None, "b" * 40), ("a" * 40, "topic"), ("main", None)])
def test_full_pr_diff_requires_full_shas_before_fetch(base, head) -> None:
    def runner(*args, **kwargs):
        raise AssertionError("invalid ranges must not be fetched")

    gh = GitHub(token="t", repository="o/r", runner=runner)
    with pytest.raises(ActionError, match="SHA is missing or invalid"):
        gh.pr_diff(7, base_sha=base, head_sha=head)


@pytest.mark.parametrize("kind", ["full-pr", "compare", "commit", "local"])
@pytest.mark.parametrize("bound", ["bytes", "files"])
def test_all_diff_sources_enforce_collection_limits(monkeypatch, kind, bound) -> None:
    from or_pr_review import github_ops as mod
    from or_pr_review.errors import InputLimitError

    # Test exact acceptance and rejection at the same shared boundary.
    monkeypatch.setattr(mod, "MAX_DIFF_BYTES", 128)
    monkeypatch.setattr(mod, "MAX_DIFF_FILES", 2)
    raw = ""
    git_calls = []

    def runner(cmd, **kwargs):
        if kind == "local":
            raise ActionError("HTTP 406: PullRequest.diff too_large")
        if "-H" not in cmd:
            return json.dumps({"status": "ahead", "behind_by": 0})
        return raw

    def git_runner(*args, **kwargs):
        git_calls.append(args)
        return raw

    gh = GitHub(
        token="t",
        repository="o/r",
        runner=runner,
        git_runner=git_runner,
        source_workspace="/checkout",
    )

    def fetch():
        if kind in {"full-pr", "local"}:
            return gh.pr_diff(7, base_sha="a" * 40, head_sha="b" * 40)
        if kind == "compare":
            return gh.compare_diff("a" * 40, "b" * 40)
        return gh.commit_diff("b" * 40)

    if bound == "bytes":
        raw = "é" * 64
    else:
        raw = "diff --git a/a b/a\n" * 2
    assert fetch() == raw
    raw += "x" if bound == "bytes" else "diff --git a/c b/c\n"
    with pytest.raises(InputLimitError, match=f"{bound[:-1] if bound == 'bytes' else 'file'}"):
        fetch()
    if kind == "full-pr":
        assert not git_calls


@pytest.mark.parametrize("method", ["pr_view", "compare", "comments"])
def test_json_payloads_are_bounded_before_parsing(monkeypatch, method) -> None:
    from or_pr_review import github_ops as mod
    from or_pr_review.errors import InputLimitError

    monkeypatch.setattr(mod, "MAX_PAYLOAD_BYTES", 20)
    gh = GitHub(token="t", repository="o/r", runner=lambda *a, **kw: "é" * 11)
    with pytest.raises(InputLimitError, match="byte collection limit"):
        if method == "pr_view":
            gh.pr_view(7)
        elif method == "compare":
            gh.compare_diff("a" * 40, "b" * 40)
        else:
            gh.list_recent_issue_comments(7)


@pytest.mark.parametrize("paginated", [False, True])
def test_list_item_limit_counts_all_items_before_filtering(monkeypatch, paginated) -> None:
    from or_pr_review import github_ops as mod
    from or_pr_review.errors import InputLimitError

    monkeypatch.setattr(mod, "MAX_LIST_ITEMS", 2)
    payload = [[{}, None]] if paginated else [{}, None]
    gh = GitHub(token="t", repository="o/r", runner=lambda *a, **kw: json.dumps(payload))
    assert gh.list_issue_comments(7) == [{}]
    if paginated:
        payload.append([{}])
    else:
        payload.append({})
    with pytest.raises(InputLimitError, match="item collection limit"):
        gh.list_issue_comments(7)


def test_injected_transport_preserves_limits_and_bounds_responses(monkeypatch) -> None:
    from or_pr_review import github_ops as mod
    from or_pr_review.errors import InputLimitError

    monkeypatch.setattr(mod, "MAX_COMMAND_BYTES", 8)
    gh = GitHub(token="t", repository="o/r", runner=lambda *a, **kw: "é" * 5)
    with pytest.raises(InputLimitError):
        gh.pr_diff(7, base_sha="a" * 40, head_sha="b" * 40)

    def runner(*args, **kwargs):
        raise InputLimitError("bounded transport")

    gh = GitHub(token="t", repository="o/r", runner=runner)
    with pytest.raises(InputLimitError, match="bounded transport"):
        gh.compare_diff("a" * 40, "b" * 40)


@pytest.mark.parametrize("stream", ["stdout", "stderr", "both"])
def test_command_capture_bounds_combined_output(monkeypatch, stream) -> None:
    from or_pr_review import github_ops as mod
    from or_pr_review.errors import InputLimitError

    monkeypatch.setattr(mod, "MAX_COMMAND_BYTES", 100_000)
    script = (
        "import sys; "
        "sys.stdout.buffer.write(b'x' * 60000); sys.stdout.flush(); "
        "sys.stderr.buffer.write(b'y' * 60000); sys.stderr.flush()"
        if stream == "both"
        else f"import sys; sys.{stream}.buffer.write(b'x' * 120000); sys.{stream}.flush()"
    )
    with pytest.raises(InputLimitError, match="byte collection limit"):
        mod._bounded_command([sys.executable, "-c", script], timeout=5, label="test")


def test_command_capture_preserves_input_and_output_at_limit(monkeypatch) -> None:
    from or_pr_review import github_ops as mod

    monkeypatch.setattr(mod, "MAX_COMMAND_BYTES", 10)
    code, out, err = mod._bounded_command(
        [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"],
        timeout=5,
        label="test",
        stdin="0123456789",
    )
    assert (code, out, err) == (0, b"0123456789", b"")


def test_command_capture_timeout_reaps_process() -> None:
    from or_pr_review.github_ops import _bounded_command

    with pytest.raises(subprocess.TimeoutExpired):
        _bounded_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=1,
            label="test",
        )
