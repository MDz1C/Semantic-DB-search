"""Column statistics and automatic role suggestions (deciding / search / display only)."""

from __future__ import annotations

from .types import ColumnProfile, Roles, Table


def profile_columns(table: Table) -> list[ColumnProfile]:
    """One profile per column, in column order, each with a suggested role and reason."""
    raise NotImplementedError


def suggest_roles(table: Table, profiles: list[ColumnProfile] | None = None) -> Roles:
    """Role for every column, from the profiles' suggestions."""
    raise NotImplementedError
