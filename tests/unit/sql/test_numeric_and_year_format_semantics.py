"""BigQuery NUMERIC precision, NUMERIC → STRING rendering and ``%E4Y`` formatting.

Three divergences from real BigQuery, each confirmed live against the emulator by an
Oracle → BigQuery data-validation run (DVT row hashes):

* a ``NUMERIC`` *type* (DDL column, ``CAST(… AS NUMERIC)``) transpiled to DuckDB's bare
  ``DECIMAL`` - ``DECIMAL(18, 3)`` - so ``1234.5678`` was stored as ``1234.568``;
* ``CAST(numeric AS STRING)`` kept the scale's trailing zeros (``'1.500'``, ``'42.000'``)
  where BigQuery prints the canonical form (``'1.5'``, ``'42'``);
* ``FORMAT_DATE`` / ``FORMAT_DATETIME`` with ``%E4Y`` reached DuckDB's ``STRFTIME``, which
  rejects the specifier outright.
"""

from __future__ import annotations

from decimal import Decimal

import duckdb
import pytest

from bqemulator.domain.result import Ok
from bqemulator.sql.translator import SQLTranslator

pytestmark = pytest.mark.unit


@pytest.fixture
def t() -> SQLTranslator:
    return SQLTranslator()


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute("SET TimeZone = 'UTC'")
    yield c
    c.close()


def _sql(t: SQLTranslator, bq_sql: str, schema: dict[str, dict[str, str]] | None = None) -> str:
    result = t.translate(bq_sql, schema=schema)
    assert isinstance(result, Ok), f"Translation failed: {result}"
    return result.value


def _one(
    t: SQLTranslator,
    conn: duckdb.DuckDBPyConnection,
    bq_sql: str,
    schema: dict[str, dict[str, str]] | None = None,
) -> tuple[object, ...]:
    row = conn.execute(_sql(t, bq_sql, schema)).fetchone()
    assert row is not None
    return row


# --- NUMERIC precision -------------------------------------------------------------------


def test_a_numeric_ddl_column_keeps_bigquerys_nine_decimal_places(
    t: SQLTranslator,
    conn: duckdb.DuckDBPyConnection,
) -> None:
    conn.execute(_sql(t, "CREATE TABLE n (x NUMERIC)"))
    conn.execute("INSERT INTO n VALUES (1234.5678), (12345678901234567890.123456789)")
    values = [r[0] for r in conn.execute("SELECT x FROM n ORDER BY x").fetchall()]
    assert values == [Decimal("1234.567800000"), Decimal("12345678901234567890.123456789")]


def test_a_cast_to_numeric_keeps_bigquerys_nine_decimal_places(
    t: SQLTranslator,
    conn: duckdb.DuckDBPyConnection,
) -> None:
    assert _one(t, conn, "SELECT CAST('1234.5678' AS NUMERIC)") == (Decimal("1234.567800000"),)


def test_an_explicit_numeric_precision_is_kept(t: SQLTranslator) -> None:
    assert "DECIMAL(10, 2)" in _sql(t, "CREATE TABLE n (x NUMERIC(10, 2))")


# --- NUMERIC -> STRING -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        ("1.50", "1.5"),
        ("42", "42"),
        ("0", "0"),
        ("-0.01", "-0.01"),
        ("100", "100"),
        ("12345678.990", "12345678.99"),
        ("1234.5678", "1234.5678"),
    ],
)
def test_a_numeric_cast_to_string_prints_bigquerys_canonical_form(
    t: SQLTranslator,
    conn: duckdb.DuckDBPyConnection,
    literal: str,
    expected: str,
) -> None:
    row = _one(t, conn, f"SELECT CAST(CAST('{literal}' AS NUMERIC) AS STRING)")
    assert row == (expected,)


def test_a_numeric_column_cast_to_string_prints_bigquerys_canonical_form(
    t: SQLTranslator,
    conn: duckdb.DuckDBPyConnection,
) -> None:
    conn.execute(_sql(t, "CREATE TABLE n (id INT64, x NUMERIC)"))
    conn.execute("INSERT INTO n VALUES (1, 1.5), (2, 42), (3, NULL)")
    sql = _sql(
        t,
        "SELECT id, CAST(x AS STRING) FROM n ORDER BY id",
        schema={"n": {"id": "BIGINT", "x": "DECIMAL(38, 9)"}},
    )
    assert conn.execute(sql).fetchall() == [(1, "1.5"), (2, "42"), (3, None)]


def test_a_string_cast_of_a_non_numeric_value_is_left_alone(
    t: SQLTranslator,
    conn: duckdb.DuckDBPyConnection,
) -> None:
    assert _one(t, conn, "SELECT CAST('100.000' AS STRING), CAST(100 AS STRING)") == (
        "100.000",
        "100",
    )


# --- %E4Y --------------------------------------------------------------------------------


def test_format_datetime_accepts_the_four_digit_year_specifier(
    t: SQLTranslator,
    conn: duckdb.DuckDBPyConnection,
) -> None:
    row = _one(
        t, conn, "SELECT FORMAT_DATETIME('%E4Y-%m-%d %H:%M:%S', DATETIME '2026-10-01 13:45:07')"
    )
    assert row == ("2026-10-01 13:45:07",)


def test_format_date_pads_a_small_year_to_four_digits_with_e4y(
    t: SQLTranslator,
    conn: duckdb.DuckDBPyConnection,
) -> None:
    assert _one(t, conn, "SELECT FORMAT_DATE('%E4Y-%m-%d', DATE '0005-01-02')") == ("0005-01-02",)


def test_an_escaped_e4y_stays_literal(t: SQLTranslator, conn: duckdb.DuckDBPyConnection) -> None:
    assert _one(t, conn, "SELECT FORMAT_DATE('%%E4Y %E4Y', DATE '2026-10-01')") == ("%E4Y 2026",)


def test_a_column_named_numeric_is_not_mistaken_for_the_type(
    t: SQLTranslator,
    conn: duckdb.DuckDBPyConnection,
) -> None:
    conn.execute(_sql(t, "CREATE TABLE n (`numeric` INT64, x NUMERIC)"))
    conn.execute("INSERT INTO n VALUES (7, 1.25)")
    assert _one(t, conn, "SELECT `numeric`, x FROM n") == (7, Decimal("1.250000000"))


def test_a_numeric_struct_field_keeps_bigquerys_precision(t: SQLTranslator) -> None:
    assert "DECIMAL(38, 9)" in _sql(t, "SELECT CAST(NULL AS STRUCT<a NUMERIC>)")
