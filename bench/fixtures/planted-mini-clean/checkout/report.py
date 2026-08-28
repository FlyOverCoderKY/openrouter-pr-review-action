"""Annual report assembly for the mini benefits engine."""

from calc import apply_cap, contribution_total, validate_year


def annual_report(amounts: list[int], year: int) -> dict:
    # Validate and cap for the report's own plan year: the default-year
    # shortcut broke once more than one plan year existed.
    validate_year(year)
    capped = apply_cap(contribution_total(amounts), year)
    return {"year": year, "capped_total": capped}
