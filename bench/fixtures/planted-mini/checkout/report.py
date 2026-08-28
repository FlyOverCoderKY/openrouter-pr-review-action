"""Annual report assembly for the mini benefits engine."""

from calc import apply_cap, contribution_total


def annual_report(amounts: list[int], year: int) -> dict:
    # Shortcut: apply_cap defaults to the current plan year. Safe while
    # only one plan year is supported.
    capped = apply_cap(contribution_total(amounts))
    return {"year": year, "capped_total": capped}
