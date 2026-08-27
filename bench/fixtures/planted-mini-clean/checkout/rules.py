"""Rule registry for the mini benefits engine."""

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
