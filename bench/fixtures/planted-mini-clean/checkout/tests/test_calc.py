from calc import apply_cap, average_contribution, contribution_total


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
