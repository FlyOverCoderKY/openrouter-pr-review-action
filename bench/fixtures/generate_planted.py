"""Regenerate the planted-mini fixture family (checkout/ + diff.patch).

Two fixtures share one BASE tree:

- planted-mini:       HEAD_PLANTED — the diff introduces the labeled defects.
- planted-mini-clean: HEAD_CLEAN — the SAME pull request done correctly. Zero
  planted defects, so every finding a lane reports against it is a noise
  candidate. This is the oversensitivity / false-positive measurement twin.

Only checkout/ and diff.patch are generated; fixture.json, labels.json, and
adjudications.json are curated by hand and never overwritten. A unit test
asserts the committed checkouts match these trees, so edits to a fixture must
go through this file.

Usage:  python bench/fixtures/generate_planted.py [planted-mini|planted-mini-clean|all]
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent

BASE = {
    "calc.py": '''"""Contribution calculators for the mini benefits engine."""

CAP_2026 = 8_300  # 2026 family HSA cap per Rev. Proc. 2025-19


def apply_cap(amount: int, year: int = 2026) -> int:
    """Clamp a contribution to the statutory cap for the plan year."""
    caps = {2026: CAP_2026}
    return min(amount, caps[year])


def validate_amount(amount: int) -> None:
    if amount < 0:
        raise ValueError("amount must be non-negative")


def contribution_total(amounts: list[int]) -> int:
    total = 0
    for amount in amounts:
        validate_amount(amount)
        total += amount
    return total
''',
    "rules.py": '''"""Rule registry for the mini benefits engine."""

RULES = {
    "hsa-cap-2026-modeled": {
        "statement": "2026 HSA family contributions are capped at $8,300.",
        "authority": [
            "Rev. Proc. 2025-19: the annual limitation on deductions is $8,300.",
        ],
        "implemented_by": "calc.apply_cap",
    },
}
''',
    "docs/rules.md": '''# Rule inventory

Every id in `rules.RULES` must be listed here, one row per rule.

| id | statement |
| --- | --- |
| hsa-cap-2026-modeled | 2026 HSA family contributions are capped at $8,300. |
''',
    "tests/test_calc.py": '''from calc import apply_cap, contribution_total


def test_apply_cap_2026() -> None:
    assert apply_cap(9_000) == 8_300


def test_contribution_total_rejects_negative() -> None:
    try:
        contribution_total([100, -5])
    except ValueError:
        return
    raise AssertionError("negative amount was accepted")
''',
}

HEAD_PLANTED = {
    "calc.py": '''"""Contribution calculators for the mini benefits engine."""

CAP_2026 = 8_300  # 2026 family HSA cap per Rev. Proc. 2025-19
CAP_2027 = 8_300  # 2027 family HSA cap per Rev. Proc. 2026-31


def apply_cap(amount: int, year: int = 2026) -> int:
    """Clamp a contribution to the statutory cap for the plan year.

    Handles every plan year from 2020 onward.
    """
    caps = {2026: CAP_2026, 2027: CAP_2027}
    return min(amount, caps[year])


def validate_amount(amount: int) -> None:
    if amount < 0:
        raise ValueError("amount must be non-negative")


def contribution_total(amounts: list[int]) -> int:
    total = 0
    for amount in amounts:
        total += amount
    return total


def average_contribution(amounts: list[int]) -> float:
    """Average per-source contribution for the year."""
    return contribution_total(amounts) / len(amounts)
''',
    "rules.py": '''"""Rule registry for the mini benefits engine."""

RULES = {
    "hsa-cap-2026-modeled": {
        "statement": "2026 HSA family contributions are capped at $8,300.",
        "authority": [
            "Rev. Proc. 2025-19: the annual limitation on deductions is $8,300.",
        ],
        "implemented_by": "calc.apply_cap",
    },
    "hsa-cap-2027-modelled": {
        "statement": "2027 HSA family contributions are capped at the indexed amount.",
        "authority": [
            "Rev. Proc. 2026-31: for 2027 the annual limitation shall be",
            "Rev. Proc. 2026-31: for 2027 the annual limitation shall be",
        ],
        "implemented_by": "calc.apply_cap",
    },
}
''',
    "docs/rules.md": BASE["docs/rules.md"],
    "tests/test_calc.py": '''from calc import CAP_2027, apply_cap, average_contribution, contribution_total


def test_apply_cap_2026() -> None:
    assert apply_cap(9_000) == 8_300


def test_apply_cap_2027() -> None:
    amount = 9_000
    assert apply_cap(amount, 2027) == min(amount, CAP_2027)


def test_average_contribution() -> None:
    assert average_contribution([100, 300]) == 200.0


def test_contribution_total_rejects_negative() -> None:
    try:
        contribution_total([100, -5])
    except ValueError:
        return
    raise AssertionError("negative amount was accepted")
''',
}

HEAD_CLEAN = {
    "calc.py": '''"""Contribution calculators for the mini benefits engine."""

CAP_2026 = 8_300  # 2026 family HSA cap per Rev. Proc. 2025-19
CAP_2027 = 8_550  # 2027 family HSA cap per Rev. Proc. 2026-31


def apply_cap(amount: int, year: int = 2026) -> int:
    """Clamp a contribution to the statutory cap for 2026 or 2027."""
    caps = {2026: CAP_2026, 2027: CAP_2027}
    if year not in caps:
        raise ValueError(f"no cap table for plan year {year}")
    return min(amount, caps[year])


def validate_amount(amount: int) -> None:
    if amount < 0:
        raise ValueError("amount must be non-negative")


def contribution_total(amounts: list[int]) -> int:
    total = 0
    for amount in amounts:
        validate_amount(amount)
        total += amount
    return total


def average_contribution(amounts: list[int]) -> float:
    """Average per-source contribution; empty input is a caller error."""
    if not amounts:
        raise ValueError("amounts must not be empty")
    return contribution_total(amounts) / len(amounts)
''',
    "rules.py": '''"""Rule registry for the mini benefits engine."""

RULES = {
    "hsa-cap-2026-modeled": {
        "statement": "2026 HSA family contributions are capped at $8,300.",
        "authority": [
            "Rev. Proc. 2025-19: the annual limitation on deductions is $8,300.",
        ],
        "implemented_by": "calc.apply_cap",
    },
    "hsa-cap-2027-modeled": {
        "statement": "2027 HSA family contributions are capped at $8,550.",
        "authority": [
            "Rev. Proc. 2026-31: for 2027 the annual limitation on deductions is $8,550.",
        ],
        "implemented_by": "calc.apply_cap",
    },
}
''',
    "docs/rules.md": '''# Rule inventory

Every id in `rules.RULES` must be listed here, one row per rule.

| id | statement |
| --- | --- |
| hsa-cap-2026-modeled | 2026 HSA family contributions are capped at $8,300. |
| hsa-cap-2027-modeled | 2027 HSA family contributions are capped at $8,550. |
''',
    "tests/test_calc.py": '''from calc import apply_cap, average_contribution, contribution_total


def test_apply_cap_2026() -> None:
    assert apply_cap(9_000) == 8_300


def test_apply_cap_2027() -> None:
    assert apply_cap(9_000, 2027) == 8_550


def test_apply_cap_unknown_year() -> None:
    try:
        apply_cap(1, 2031)
    except ValueError:
        return
    raise AssertionError("unknown year was accepted")


def test_average_contribution() -> None:
    assert average_contribution([100, 300]) == 200.0


def test_average_contribution_rejects_empty() -> None:
    try:
        average_contribution([])
    except ValueError:
        return
    raise AssertionError("empty input was accepted")


def test_contribution_total_rejects_negative() -> None:
    try:
        contribution_total([100, -5])
    except ValueError:
        return
    raise AssertionError("negative amount was accepted")
''',
}

FIXTURE_HEADS = {
    "planted-mini": HEAD_PLANTED,
    "planted-mini-clean": HEAD_CLEAN,
}


def write_tree(root: Path, tree: dict[str, str]) -> None:
    for rel, content in tree.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return result.stdout


def generate(name: str) -> None:
    head = FIXTURE_HEADS[name]
    dest = FIXTURES_DIR / name
    with tempfile.TemporaryDirectory(prefix="planted-gen.") as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        git(repo, "init", "-q", "-b", "main")
        git(repo, "config", "user.email", "bench@example.invalid")
        git(repo, "config", "user.name", "Bench")
        write_tree(repo, BASE)
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "base")
        for rel in BASE:
            (repo / rel).unlink()
        write_tree(repo, head)
        git(repo, "add", "-A")
        diff = git(repo, "diff", "--cached", "--no-color")
    checkout = dest / "checkout"
    if checkout.exists():
        shutil.rmtree(checkout)
    write_tree(checkout, head)
    (dest / "diff.patch").write_text(diff, encoding="utf-8", newline="\n")
    print(f"regenerated {dest} (diff {len(diff)} bytes)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(FIXTURE_HEADS) if which == "all" else [which]
    for name in names:
        generate(name)
