"""Unit tests for orius.utils.sql safety validators."""

from __future__ import annotations

import pytest

from orius.utils.sql import ALLOWED_COLUMN_TYPES, validate_column_type, validate_sql_identifier


class TestValidateSqlIdentifier:
    def test_simple_name(self):
        assert validate_sql_identifier("dispatch_certificates") == "dispatch_certificates"

    def test_leading_underscore(self):
        assert validate_sql_identifier("_private") == "_private"

    def test_mixed_case_and_digits(self):
        assert validate_sql_identifier("table_v2") == "table_v2"

    def test_returns_the_name(self):
        name = "my_table"
        assert validate_sql_identifier(name) is name

    @pytest.mark.parametrize(
        "bad",
        [
            "a; DROP TABLE users--",
            "table name",          # space
            "1starts_with_digit",  # leading digit
            "",                    # empty
            "col'injection",       # single quote
            'col"injection',       # double quote
            "col-name",            # hyphen
            "col.name",            # dot
            "col\x00null",         # null byte
        ],
    )
    def test_rejects_unsafe(self, bad: str):
        with pytest.raises(ValueError, match="Unsafe SQL"):
            validate_sql_identifier(bad)

    def test_custom_label_appears_in_error(self):
        with pytest.raises(ValueError, match="table name"):
            validate_sql_identifier("bad name", label="table name")


class TestValidateColumnType:
    @pytest.mark.parametrize("t", sorted(ALLOWED_COLUMN_TYPES))
    def test_all_allowed_types_pass(self, t: str):
        assert validate_column_type(t) == t

    def test_returns_the_type(self):
        ct = "VARCHAR"
        assert validate_column_type(ct) is ct

    @pytest.mark.parametrize(
        "bad",
        [
            "varchar",              # wrong case
            "VARCHAR(255)",         # length suffix not in allowlist
            "INT",                  # alias not in allowlist
            "TEXT[]",               # array modifier
            "BOOLEAN; DROP TABLE",  # injection attempt
            "",
        ],
    )
    def test_rejects_unknown_types(self, bad: str):
        with pytest.raises(ValueError, match="Unsupported SQL column type"):
            validate_column_type(bad)
