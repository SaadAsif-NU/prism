"""Tests for scalar functions and the AggregateExpr node."""

from __future__ import annotations

import pytest

from prism import Database, Table, col
from prism.aggregate import AggFunc, AggregateExpr
from prism.types import DataType


class TestScalarKernels:
    def test_abs(self) -> None:
        db = Database()
        db.register("t", Table.from_pydict({"x": [-3, 4]}, types={"x": DataType.INTEGER}))
        assert db.sql("SELECT ABS(x) AS a FROM t").column("a").to_pylist() == [3, 4]

    def test_round_default_digits(self) -> None:
        db = Database()
        db.register("t", Table.from_pydict({"x": [3.14159]}, types={"x": DataType.FLOAT}))
        assert db.sql("SELECT ROUND(x) AS r FROM t").column("r").to_pylist() == [3.0]

    def test_coalesce_picks_first_non_null(self) -> None:
        db = Database()
        db.register(
            "t",
            Table.from_pydict(
                {"a": [None, 2], "b": [10, 20]},
                types={"a": DataType.INTEGER, "b": DataType.INTEGER},
            ),
        )
        assert db.sql("SELECT COALESCE(a, b) AS c FROM t").column("c").to_pylist() == [10, 2]

    def test_length_preserves_null(self) -> None:
        db = Database()
        db.register("t", Table.from_pydict({"s": ["ab", None]}, types={"s": DataType.TEXT}))
        assert db.sql("SELECT LENGTH(s) AS n FROM t").column("n").to_pylist() == [2, None]

    def test_arity_error(self) -> None:
        db = Database()
        db.register("t", Table.from_pydict({"x": [1]}, types={"x": DataType.INTEGER}))
        with pytest.raises(TypeError, match="argument"):
            db.sql("SELECT UPPER(x, x) FROM t")

    def test_abs_on_text_errors(self) -> None:
        db = Database()
        db.register("t", Table.from_pydict({"s": ["x"]}, types={"s": DataType.TEXT}))
        with pytest.raises(TypeError):
            db.sql("SELECT ABS(s) FROM t")


class TestDistinctReducers:
    def _db(self) -> Database:
        db = Database()
        db.register(
            "t",
            Table.from_pydict({"v": [1, 1, 2, 3, 3]}, types={"v": DataType.INTEGER}),
        )
        return db

    def test_sum_distinct(self) -> None:
        assert self._db().sql("SELECT SUM(DISTINCT v) AS s FROM t").column("s").to_pylist() == [6]

    def test_avg_distinct(self) -> None:
        assert self._db().sql("SELECT AVG(DISTINCT v) AS a FROM t").column("a").to_pylist() == [2.0]

    def test_min_max_distinct(self) -> None:
        result = self._db().sql("SELECT MIN(DISTINCT v) AS lo, MAX(DISTINCT v) AS hi FROM t")
        assert result.to_rows() == [(1, 3)]

    def test_distinct_over_empty_is_null(self) -> None:
        db = Database()
        db.register("t", Table.from_pydict({"v": [None]}, types={"v": DataType.INTEGER}))
        assert db.sql("SELECT SUM(DISTINCT v) AS s FROM t").column("s").to_pylist() == [None]


class TestAggregateExpr:
    def test_output_names(self) -> None:
        assert AggregateExpr(AggFunc.COUNT, None).output_name() == "COUNT(*)"
        assert AggregateExpr(AggFunc.SUM, col("x")).output_name() == "SUM(x)"
        assert (
            AggregateExpr(AggFunc.MAX, col("x"), distinct=True).output_name() == "MAX(DISTINCT x)"
        )

    def test_resolve_type(self) -> None:
        schema = {"x": DataType.INTEGER}
        assert AggregateExpr(AggFunc.COUNT, None).resolve_type(schema) is DataType.INTEGER
        assert AggregateExpr(AggFunc.AVG, col("x")).resolve_type(schema) is DataType.FLOAT
        assert AggregateExpr(AggFunc.MAX, col("x")).resolve_type(schema) is DataType.INTEGER

    def test_references(self) -> None:
        assert AggregateExpr(AggFunc.SUM, col("salary")).references() == {"salary"}
        assert AggregateExpr(AggFunc.COUNT, None).references() == set()

    def test_evaluate_raises(self) -> None:
        t = Table.from_pydict({"x": [1, 2]})
        with pytest.raises(TypeError, match="aggregate"):
            AggregateExpr(AggFunc.SUM, col("x")).evaluate(t)

    def test_repr(self) -> None:
        assert repr(AggregateExpr(AggFunc.COUNT, None)) == "COUNT(*)"
