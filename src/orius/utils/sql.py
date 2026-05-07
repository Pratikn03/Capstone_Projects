"""SQL safety utilities."""

from __future__ import annotations

import re

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

ALLOWED_COLUMN_TYPES: frozenset[str] = frozenset(
    {"BOOLEAN", "VARCHAR", "DOUBLE", "INTEGER", "BIGINT", "FLOAT", "REAL", "TEXT", "TIMESTAMP"}
)


def validate_sql_identifier(name: str, label: str = "identifier") -> str:
    """Raise ValueError if *name* is not a safe SQL identifier, else return it."""
    if not _SAFE_IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Unsafe SQL {label} {name!r}: only letters, digits, and underscores are allowed"
        )
    return name


def validate_column_type(column_type: str) -> str:
    """Raise ValueError if *column_type* is not in the known-safe allowlist, else return it."""
    if column_type not in ALLOWED_COLUMN_TYPES:
        raise ValueError(
            f"Unsupported SQL column type {column_type!r}. "
            f"Allowed: {sorted(ALLOWED_COLUMN_TYPES)}"
        )
    return column_type
