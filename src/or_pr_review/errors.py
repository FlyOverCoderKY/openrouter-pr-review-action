"""Action vs lane failure classes.

Operational and schema errors fail the job. Lane failures are recorded and
the judge continues (fail-open) on whatever structured results arrived.
"""

from __future__ import annotations


class ActionError(Exception):
    """Operational error of the action itself. Fail the job."""


class SchemaError(ActionError):
    """Lane-artifact or internal contract mismatch. Fail-closed."""


class LaneError(Exception):
    """One model lane failed. Fail-open that lane only."""
