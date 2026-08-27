from calc import CAP_2027, apply_cap, average_contribution, contribution_total


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
