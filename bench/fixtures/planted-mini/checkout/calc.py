"""Contribution calculators for the mini benefits engine."""

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
