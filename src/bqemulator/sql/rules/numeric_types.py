"""Translation rules for BigQuery NUMERIC / BIGNUMERIC parse helpers.

* ``PARSE_NUMERIC(s)`` (ADR 0023 §1.B) — BigQuery returns ``NUMERIC``
  for any STRING that parses as a decimal in the ``DECIMAL(38, 9)``
  range. DuckDB ships no ``PARSE_NUMERIC`` builtin; the equivalent
  expression is ``CAST(s AS DECIMAL(38, 9))``. The cast handles every
  literal-or-column input we exercise in the corpus.

* ``PARSE_BIGNUMERIC(s)`` (ADR 0023 §1.B) — BigQuery returns
  ``BIGNUMERIC``. We route through the ``bqemu_to_bignumeric`` Python
  UDF (see :mod:`bqemulator.sql.builtin_udfs`) which is registered with
  return type ``DECIMAL(38, 10)`` so the REST schema renderer surfaces
  the column as BIGNUMERIC (any DECIMAL whose ``scale > 9`` is BIGNUMERIC
  per ADR 0023 §1.B). Values exceeding DECIMAL(38, …)'s 38-digit cap
  cannot be represented and cascade to ADR 0023 §1.I (bit-exact value
  mismatch).

The rules match the typed SQLGlot nodes ``exp.ParseJSON``-style — for
``PARSE_NUMERIC`` / ``PARSE_BIGNUMERIC`` SQLGlot does not synthesise a
typed node, so both rules match by anonymous-name lookup.
"""

from __future__ import annotations

from sqlglot import exp

from bqemulator.sql.rules import register
from bqemulator.sql.rules._base import TranslationRule

_NUMERIC_PRECISION = 38
_NUMERIC_SCALE = 9


@register
class ParseNumericRule(TranslationRule):
    """``PARSE_NUMERIC(s)`` → ``CAST(s AS DECIMAL(38, 9))``.

    SQLGlot parses ``PARSE_NUMERIC`` into a typed :class:`exp.ParseNumeric`
    node (not :class:`exp.Anonymous`), so the rule matches by type-name
    rather than the anonymous-function dispatch. The single operand
    lives in ``node.this``.
    """

    name = "PARSE_NUMERIC"

    def applies_to(self, node: exp.Expression) -> bool:
        """Match the typed ``ParseNumeric`` node."""
        return type(node).__name__ == "ParseNumeric"

    def rewrite(self, node: exp.Expression) -> exp.Expression:
        """Emit ``CAST(arg AS DECIMAL(38, 9))``."""
        arg = node.this
        if arg is None:
            return node
        return exp.Cast(
            this=arg.copy(),
            to=exp.DataType.build(f"DECIMAL({_NUMERIC_PRECISION}, {_NUMERIC_SCALE})"),
        )


@register
class ParseBignumericRule(TranslationRule):
    """``PARSE_BIGNUMERIC(s)`` → ``bqemu_to_bignumeric(s)``.

    The UDF is registered at engine startup with return type
    ``DECIMAL(38, 10)`` — the scale of 10 (> 9) is the marker the REST
    schema renderer uses to surface BIGNUMERIC. SQLGlot parses
    ``PARSE_BIGNUMERIC`` into a typed :class:`exp.ParseBignumeric` node;
    the rule matches by type-name and reads the single operand from
    ``node.this``.
    """

    name = "PARSE_BIGNUMERIC"

    def applies_to(self, node: exp.Expression) -> bool:
        """Match the typed ``ParseBignumeric`` node."""
        return type(node).__name__ == "ParseBignumeric"

    def rewrite(self, node: exp.Expression) -> exp.Expression:
        """Emit ``bqemu_to_bignumeric(arg)``."""
        arg = node.this
        if arg is None:
            return node
        return exp.Anonymous(this="bqemu_to_bignumeric", expressions=[arg.copy()])


def _is_decimal(node: exp.Expression | None) -> bool:
    """Return whether an expression is already known to be ``DECIMAL``.

    That is a ``CAST(… AS DECIMAL)``, or any node ``annotate_types`` typed as one (a NUMERIC
    column, arithmetic over NUMERICs).
    """
    if node is None:
        return False
    if isinstance(node, exp.Cast) and node.to is not None:
        return node.to.is_type(exp.DataType.Type.DECIMAL)
    node_type = node.type
    return isinstance(node_type, exp.DataType) and node_type.is_type(exp.DataType.Type.DECIMAL)


# BigQuery prints a NUMERIC in its shortest exact form; DuckDB pads to the column's scale
# (``1.500``). Strip a fractional part's trailing zeros, and the point itself when nothing
# but zeros follows it: ``1.500`` → ``1.5``, ``42.000`` → ``42``, ``100`` stays ``100``.
_TRAILING_ZEROS = r"(\.[0-9]*[1-9])0+$|\.0+$"


@register
class NumericToStringRule(TranslationRule):
    """``CAST(<NUMERIC> AS STRING)`` → BigQuery's canonical rendering, no padded scale.

    Only fires for an operand known to be ``DECIMAL`` (see :func:`_is_decimal`) - a STRING
    that merely looks numeric (``CAST('100.000' AS STRING)``) is never rewritten.
    """

    name = "NUMERIC_TO_STRING"

    def applies_to(self, node: exp.Expression) -> bool:
        """Match ``CAST``/``TRY_CAST`` to a text type over a ``DECIMAL`` operand."""
        if not isinstance(node, exp.Cast) or node.to is None:
            return False
        if not (
            node.to.is_type(exp.DataType.Type.VARCHAR) or node.to.is_type(exp.DataType.Type.TEXT)
        ):
            return False
        return _is_decimal(node.this)

    def rewrite(self, node: exp.Expression) -> exp.Expression:
        r"""Emit ``REGEXP_REPLACE(CAST(x AS TEXT), <trailing zeros>, '\1')``."""
        return exp.Anonymous(
            this="REGEXP_REPLACE",
            expressions=[
                node.copy(),
                exp.Literal.string(_TRAILING_ZEROS),
                exp.Literal.string("\\1"),
            ],
        )


__all__ = [
    "NumericToStringRule",
    "ParseBignumericRule",
    "ParseNumericRule",
]
