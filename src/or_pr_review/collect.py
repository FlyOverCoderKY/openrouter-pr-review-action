"""Collect PR metadata and the scoped diff.

latest-commit never silently falls back to the full PR diff. If the
before...after range cannot be used, the single latest head commit is
embedded and a visible notice is recorded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol

from or_pr_review.errors import ActionError, DivergedRangeError

ScopeName = Literal["full-pr", "latest-commit"]
ReviewMode = Literal["auto", "initial", "verify"]
ResolvedMode = Literal["initial", "verify"]
DiffKind = Literal["full-pr", "commit-range", "single-commit"]

_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
_ZERO_RE = re.compile(r"^0+$")

MISSING_BEFORE_NOTICE = (
    "The before SHA was missing (first push, force-push, or workflow_dispatch). "
    "This prompt embeds only the single latest commit on the PR head. "
    "This is not a full-PR review and the full pull request diff was not fetched."
)

COMPARE_FAILED_NOTICE = (
    "The before...after compare failed (a transient GitHub error). "
    "This prompt embeds only the single latest commit on the PR head. "
    "This is not a full-PR review and the full pull request diff was not fetched."
)

DIVERGED_NOTICE = (
    "The before...after range is not a linear fast-forward: history was "
    "rewritten (force-push) and the earlier commit is no longer an ancestor. "
    "This prompt embeds only the single latest commit on the PR head. "
    "This is not a full-PR review and the full pull request diff was not fetched."
)


class ReviewSource(Protocol):
    def pr_view(self, number: int) -> dict[str, object]: ...

    def pr_diff(self, number: int) -> str: ...

    def compare_diff(self, before: str, after: str) -> str: ...

    def commit_diff(self, sha: str) -> str: ...


@dataclass(frozen=True)
class DiffPlan:
    scope: ScopeName
    kind: DiffKind
    from_sha: str | None
    to_sha: str | None
    fallback_notice: str | None


@dataclass(frozen=True)
class Truncation:
    text: str
    truncated: bool
    original_bytes: int
    embedded_bytes: int
    max_diff_kb: int

    @property
    def notice(self) -> str | None:
        if not self.truncated:
            return None
        return (
            f"Diff truncated from {self.original_bytes / 1024:.1f} KB to "
            f"{self.embedded_bytes / 1024:.1f} KB (max_diff_kb={self.max_diff_kb}). "
            "Later files/hunks are missing. This review is a partial verdict and "
            "must not be treated as clean."
        )


@dataclass(frozen=True)
class CollectedReview:
    pr_number: int
    title: str
    body: str
    head_sha: str
    base_ref: str
    head_ref: str
    plan: DiffPlan
    truncation: Truncation
    mode: ResolvedMode
    # Changed paths from the FULL collected diff, before the embed cap:
    # consumers that reason about "what changed on this PR" (path profiles)
    # must not be blinded by byte truncation of the prompt embed.
    all_changed_paths: tuple[str, ...] = ()

    @property
    def diff(self) -> str:
        return self.truncation.text


def parse_scope(value: str) -> ScopeName:
    scope = (value or "").strip().lower()
    if scope in {"full-pr", "latest-commit"}:
        return scope  # type: ignore[return-value]
    raise ActionError("review_scope must be 'full-pr' or 'latest-commit'")


def parse_mode(value: str) -> ReviewMode:
    mode = (value or "").strip().lower()
    if mode in {"auto", "initial", "verify"}:
        return mode  # type: ignore[return-value]
    raise ActionError("review_mode must be 'auto', 'initial', or 'verify'")


def resolve_mode(mode: ReviewMode, event_action: str | None) -> ResolvedMode:
    if mode == "initial":
        return "initial"
    if mode == "verify":
        return "verify"
    action = (event_action or "").strip().lower()
    if action == "synchronize":
        return "verify"
    return "initial"


def head_sha_from_pr(pr: dict[str, object]) -> str | None:
    """Normalized head SHA from a PR payload (headRefOid or nested head.sha)."""
    return normalize_sha(_as_str(pr.get("headRefOid")) or _nested_head(pr))


def normalize_sha(value: str | None) -> str | None:
    if value is None:
        return None
    sha = value.strip()
    if sha == "" or _ZERO_RE.fullmatch(sha) or not _SHA_RE.fullmatch(sha):
        return None
    return sha.lower()


def plan_diff(
    *,
    scope: ScopeName,
    before_sha: str | None,
    after_sha: str | None,
    head_sha: str | None,
) -> DiffPlan:
    """Decide which range to embed. latest-commit never plans a full-PR diff."""
    head = normalize_sha(head_sha)
    after = normalize_sha(after_sha) or head
    before = normalize_sha(before_sha)

    if scope == "full-pr":
        return DiffPlan(
            scope=scope,
            kind="full-pr",
            from_sha=None,
            to_sha=after,
            fallback_notice=None,
        )

    if before and after and before != after:
        return DiffPlan(
            scope=scope,
            kind="commit-range",
            from_sha=before,
            to_sha=after,
            fallback_notice=None,
        )

    if after is None:
        raise ActionError(
            "latest-commit requires a head SHA (github.event.after or "
            "pull_request.head.sha). Refusing to fall back to the full PR diff."
        )

    return DiffPlan(
        scope=scope,
        kind="single-commit",
        from_sha=None,
        to_sha=after,
        fallback_notice=MISSING_BEFORE_NOTICE,
    )


def truncate_diff(diff: str, max_diff_kb: int) -> Truncation:
    if max_diff_kb <= 0:
        raise ActionError("max_diff_kb must be a positive integer")
    limit = max_diff_kb * 1024
    data = diff.encode("utf-8")
    if len(data) <= limit:
        return Truncation(
            text=diff,
            truncated=False,
            original_bytes=len(data),
            embedded_bytes=len(data),
            max_diff_kb=max_diff_kb,
        )
    cut = _cut_at_boundary(data, limit)
    text = cut.decode("utf-8", errors="ignore")
    return Truncation(
        text=text,
        truncated=True,
        original_bytes=len(data),
        embedded_bytes=len(text.encode("utf-8")),
        max_diff_kb=max_diff_kb,
    )


def _cut_at_boundary(data: bytes, limit: int) -> bytes:
    prefix = data[:limit]
    minimum = limit // 2
    boundary = max(prefix.rfind(b"\ndiff --git "), prefix.rfind(b"\n@@ "))
    if boundary + 1 > minimum:
        return prefix[: boundary + 1]
    newline = prefix.rfind(b"\n")
    if newline + 1 > minimum:
        return prefix[: newline + 1]
    return prefix


def fetch_scoped_diff(pr_number: int, plan: DiffPlan, source: ReviewSource) -> tuple[str, DiffPlan]:
    """Return (diff, plan). latest-commit never calls pr_diff."""
    if plan.kind == "full-pr":
        return source.pr_diff(pr_number), plan

    if plan.kind == "commit-range":
        if plan.from_sha is None or plan.to_sha is None:
            raise ActionError("commit-range plan is missing SHAs")
        try:
            return source.compare_diff(plan.from_sha, plan.to_sha), plan
        except DivergedRangeError:
            # Only a genuine non-fast-forward carries the diverged notice;
            # the review loop keys its reset on it.
            notice = DIVERGED_NOTICE
        except ActionError:
            notice = COMPARE_FAILED_NOTICE
        if not plan.to_sha:
            raise ActionError(
                "latest-commit fallback is missing a head SHA; "
                "refusing to fall back to the full PR diff"
            ) from None
        fallback = DiffPlan(
            scope=plan.scope,
            kind="single-commit",
            from_sha=None,
            to_sha=plan.to_sha,
            fallback_notice=notice,
        )
        return source.commit_diff(fallback.to_sha), fallback

    if not plan.to_sha:
        raise ActionError(
            "latest-commit is missing a head SHA; refusing to fall back to the full PR diff"
        )
    return source.commit_diff(plan.to_sha), plan


def collect_review(
    *,
    pr_number: int,
    scope: ScopeName,
    mode: ResolvedMode,
    before_sha: str | None,
    after_sha: str | None,
    head_sha: str | None,
    max_diff_kb: int,
    source: ReviewSource,
) -> CollectedReview:
    if mode == "initial" and scope != "full-pr":
        raise ActionError("initial review_mode requires review_scope=full-pr")

    pr = source.pr_view(pr_number)
    head_from_pr = head_sha_from_pr(pr)
    if scope == "full-pr":
        plan = plan_diff(
            scope=scope,
            before_sha=None,
            after_sha=None,
            head_sha=head_from_pr or head_sha,
        )
    else:
        plan = plan_diff(
            scope=scope,
            before_sha=before_sha,
            after_sha=after_sha,
            head_sha=head_sha or head_from_pr,
        )

    raw, plan = fetch_scoped_diff(pr_number, plan, source)
    if scope == "full-pr":
        confirmed = source.pr_view(pr_number)
        confirmed_head = head_sha_from_pr(confirmed)
        if confirmed_head is None:
            raise ActionError("PR head SHA is missing from the PR metadata; retry the review")
        if plan.to_sha and confirmed_head != plan.to_sha:
            raise ActionError("PR head changed while collecting the full-PR diff; retry the review")
        pr = confirmed
        head_from_pr = confirmed_head

    resolved_head = plan.to_sha or head_from_pr or normalize_sha(head_sha)
    if not resolved_head:
        raise ActionError("could not resolve the reviewed commit SHA")

    return CollectedReview(
        pr_number=pr_number,
        title=_as_str(pr.get("title")) or "",
        body=_as_str(pr.get("body")) or "",
        head_sha=resolved_head,
        base_ref=_as_str(pr.get("baseRefName")) or "",
        head_ref=_as_str(pr.get("headRefName")) or "",
        plan=plan,
        truncation=truncate_diff(raw, max_diff_kb),
        mode=mode,
        all_changed_paths=_all_changed_paths(raw),
    )


def _all_changed_paths(diff: str) -> tuple[str, ...]:
    # Local import: prompt imports CollectedReview from this module.
    from or_pr_review.prompt import changed_paths_from_diff

    return tuple(changed_paths_from_diff(diff))


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _nested_head(pr: dict[str, object]) -> str | None:
    head = pr.get("head")
    if isinstance(head, dict):
        sha = head.get("sha")
        if isinstance(sha, str):
            return sha
    return None
