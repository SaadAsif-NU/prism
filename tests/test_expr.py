"""Tests for the vectorised expression engine, including null semantics."""

from __future__ import annotations

import pytest

from prism import Table, col, lit
from prism.expr import Literal, schema_of
from prism.types import DataType


def _eval(expr, table: Table) -> list:  # type: ignore[no-untyped-def]
    return expr.evaluate(table).to_pylist()


class TestBasics:
    def test_column_ref(self) -> None:
        t = Table.from_pydict({"a": [1, 2, 3]})
        assert _eval(col("a"), t) == [1, 2, 3]

    def test_literal_broadcast(self) -> None:
        t = Table.from_pydict({"a": [1, 2, 3]})
        assert _eval(lit(7), t) == [7, 7, 7]

    def test_null_literal(self) -> None:
        t = Table.from_pydict({"a": [1, 2]})
        assert _eval(lit(None), t) == [None, None]

    def test_alias_renames(self) -> None:
        t = Table.from_pydict({"a": [1, 2]})
        result = (col("a") + 1).alias("b").evaluate(t)
        assert result.name == "b"


class TestArithmetic:
    def test_add(self) -> None:
        t = Table.from_pydict({"a": [1, 2], "b": [10, 20]})
        assert _eval(col("a") + col("b"), t) == [11, 22]

    def test_subtract_and_multiply(self) -> None:
        t = Table.from_pydict({"a": [5, 6]})
        assert _eval(col("a") - 1, t) == [4, 5]
        assert _eval(col("a") * 2, t) == [10, 12]

    def test_integer_stays_integer(self) -> None:
        t = Table.from_pydict({"a": [4, 6]})
        result = (col("a") + 1).evaluate(t)
        assert result.dtype is DataType.INTEGER

    def test_division_is_float(self) -> None:
        t = Table.from_pydict({"a": [7, 8]})
        result = (col("a") / 2).evaluate(t)
        assert result.dtype is DataType.FLOAT
        assert result.to_pylist() == [3.5, 4.0]

    def test_division_by_zero_is_null(self) -> None:
        t = Table.from_pydict({"a": [1, 2], "b": [0, 2]})
        assert _eval(col("a") / col("b"), t) == [None, 1.0]

    def test_modulo(self) -> None:
        t = Table.from_pydict({"a": [7, 8]})
        assert _eval(col("a") % 3, t) == [1, 2]

    def test_modulo_by_zero_is_null(self) -> None:
        t = Table.from_pydict({"a": [7], "b": [0]})
        assert _eval(col("a") % col("b"), t) == [None]

    def test_null_propagates(self) -> None:
        t = Table.from_pydict({"a": [1, None, 3]}, types={"a": DataType.INTEGER})
        assert _eval(col("a") + 1, t) == [2, None, 4]

    def test_negate(self) -> None:
        t = Table.from_pydict({"a": [1, -2]})
        assert _eval(-col("a"), t) == [-1, 2]

    def test_arithmetic_on_text_raises(self) -> None:
        t = Table.from_pydict({"a": ["x"]})
        with pytest.raises(TypeError):
            (col("a") + 1).resolve_type(schema_of(t))


class TestComparison:
    def test_equal(self) -> None:
        t = Table.from_pydict({"a": [1, 2, 3]})
        assert _eval(col("a") == 2, t) == [False, True, False]

    def test_not_equal(self) -> None:
        t = Table.from_pydict({"a": [1, 2]})
        assert _eval(col("a") != 1, t) == [False, True]

    def test_ordering(self) -> None:
        t = Table.from_pydict({"a": [1, 2, 3]})
        assert _eval(col("a") > 1, t) == [False, True, True]
        assert _eval(col("a") <= 2, t) == [True, True, False]
        assert _eval(col("a") >= 3, t) == [False, False, True]
        assert _eval(col("a") < 2, t) == [True, False, False]

    def test_text_comparison(self) -> None:
        t = Table.from_pydict({"a": ["apple", "pear"]})
        assert _eval(col("a") == "pear", t) == [False, True]
        assert _eval(col("a") < "banana", t) == [True, False]

    def test_comparison_null_propagates(self) -> None:
        t = Table.from_pydict({"a": [1, None, 3]}, types={"a": DataType.INTEGER})
        assert _eval(col("a") > 1, t) == [False, None, True]

    def test_comparison_type_is_boolean(self) -> None:
        t = Table.from_pydict({"a": [1]})
        assert (col("a") > 0).resolve_type(schema_of(t)) is DataType.BOOLEAN


class TestThreeValuedLogic:
    def _bool_table(self) -> Table:
        # All nine combinations of {True, False, None} x {True, False, None}.
        left = [True, True, True, False, False, False, None, None, None]
        right = [True, False, None, True, False, None, True, False, None]
        return Table.from_pydict(
            {"l": left, "r": right},
            types={"l": DataType.BOOLEAN, "r": DataType.BOOLEAN},
        )

    def test_and(self) -> None:
        t = self._bool_table()
        # T&T=T, T&F=F, T&N=N, F&*=F, N&T=N, N&F=F, N&N=N
        assert _eval(col("l") & col("r"), t) == [
            True,
            False,
            None,
            False,
            False,
            False,
            None,
            False,
            None,
        ]

    def test_or(self) -> None:
        t = self._bool_table()
        # T|*=T, F|T=T, F|F=F, F|N=N, N|T=T, N|F=N, N|N=N
        assert _eval(col("l") | col("r"), t) == [
            True,
            True,
            True,
            True,
            False,
            None,
            True,
            None,
            None,
        ]

    def test_not(self) -> None:
        t = Table.from_pydict({"a": [True, False, None]}, types={"a": DataType.BOOLEAN})
        assert _eval(~col("a"), t) == [False, True, None]


class TestNullChecks:
    def test_is_null(self) -> None:
        t = Table.from_pydict({"a": [1, None, 3]}, types={"a": DataType.INTEGER})
        assert _eval(col("a").is_null(), t) == [False, True, False]

    def test_is_not_null(self) -> None:
        t = Table.from_pydict({"a": [1, None, 3]}, types={"a": DataType.INTEGER})
        assert _eval(col("a").is_not_null(), t) == [True, False, True]

    def test_is_null_never_null(self) -> None:
        t = Table.from_pydict({"a": [None, None]}, types={"a": DataType.INTEGER})
        result = col("a").is_null().evaluate(t)
        assert result.null_count == 0


class TestMetadata:
    def test_references(self) -> None:
        expr = (col("a") + col("b")) > col("c")
        assert expr.references() == {"a", "b", "c"}

    def test_literal_has_no_references(self) -> None:
        assert lit(5).references() == set()

    def test_output_name(self) -> None:
        assert col("age").output_name() == "age"
        assert (col("a") + lit(1)).output_name() == "(a + 1)"

    def test_resolve_unknown_column(self) -> None:
        with pytest.raises(KeyError):
            col("missing").resolve_type({"a": DataType.INTEGER})

    def test_literal_type_inference(self) -> None:
        assert Literal(True).dtype is DataType.BOOLEAN
        assert Literal(3).dtype is DataType.INTEGER
        assert Literal(3.5).dtype is DataType.FLOAT
        assert Literal("x").dtype is DataType.TEXT
        assert Literal(None).dtype is DataType.NULL
