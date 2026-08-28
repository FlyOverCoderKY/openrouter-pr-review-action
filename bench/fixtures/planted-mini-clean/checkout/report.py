"""Annual report assembly for the mini benefits engine."""

from calc import apply_cap, contribution_total, validate_year


def annual_report(amounts: list[int], year: int) -> dict:
    validate_year(year)
    # Cap for the report's own plan year: the default-year shortcut broke
    # once more than one plan year existed.
    capped = apply_cap(contribution_total(amounts), year)
    return {"year": year, "capped_total": capped}
