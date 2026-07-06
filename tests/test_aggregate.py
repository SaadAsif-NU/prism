"""Unit tests for the aggregation operators and reducers."""

from __future__ import annotations

from prism import Table, col
from prism.aggregate import AggFunc, AggregateSpec
from prism.exec.aggregate import Distinct, HashAggregate, factorize
from prism.exec.operators import Scan
from prism.types import DataType


def _table() -> Table:
    return Table.from_pydict(
        {
            "g": ["a", "b", "a", "b", "a"],
            "x": [1, 2, 3, None, 5],
        },
        types={"g": DataType.TEXT, "x": DataType.INTEGER},
    )


def _spec(
    func: AggFunc, arg, name: str, out_type: DataType, distinct: bool = False
) -> AggregateSpec:
    return AggregateSpec(func, arg, distinct, name, out_type)


class TestFactorize:
    def test_group_ids_and_first_indices(self) -> None:
        cols = [_table().column("g")]
        group_ids, n_groups, first = factorize(cols)
        assert n_groups == 2
        assert list(group_ids) == [0, 1, 0, 1, 0]
        assert list(first) == [0, 1]

    def test_null_keys_group_together(self) -> None:
        col_ = Table.from_pydict({"g": [None, None, "x"]}).column("g")
        _, n_groups, _ = factorize([col_])
        assert n_groups == 2


class TestHashAggregate:
    def test_grouped_count_and_sum(self) -> None:
        specs = [
            _spec(AggFunc.COUNT, None, "n", DataType.INTEGER),
            _spec(AggFunc.SUM, col("x"), "s", DataType.INTEGER),
        ]
        op = HashAggregate(Scan(_table()), [col("g").alias("g")], specs)
        result = op.execute()
        by_group = dict(
            zip(result.column("g").to_pylist(), result.column("s").to_pylist(), strict=True)
        )
        assert by_group == {"a": 9, "b": 2}  # b has one NULL, summed as 2

    def test_avg_skips_nulls(self) -> None:
        specs = [_spec(AggFunc.AVG, col("x"), "a", DataType.FLOAT)]
        op = HashAggregate(Scan(_table()), [col("g").alias("g")], specs)
        result = op.execute()
        avgs = dict(
            zip(result.column("g").to_pylist(), result.column("a").to_pylist(), strict=True)
        )
        assert avgs["a"] == 3.0  # (1+3+5)/3
        assert avgs["b"] == 2.0  # only the non-null 2

    def test_min_max(self) -> None:
        specs = [
            _spec(AggFunc.MIN, col("x"), "lo", DataType.INTEGER),
            _spec(AggFunc.MAX, col("x"), "hi", DataType.INTEGER),
        ]
        result = HashAggregate(Scan(_table()), [col("g").alias("g")], specs).execute()
        lo = dict(zip(result.column("g").to_pylist(), result.column("lo").to_pylist(), strict=True))
        assert lo == {"a": 1, "b": 2}

    def test_text_min_max(self) -> None:
        t = Table.from_pydict({"g": ["x", "x"], "w": ["pear", "apple"]})
        specs = [_spec(AggFunc.MIN, col("w"), "m", DataType.TEXT)]
        result = HashAggregate(Scan(t), [col("g").alias("g")], specs).execute()
        assert result.column("m").to_pylist() == ["apple"]

    def test_global_aggregate_one_row(self) -> None:
        specs = [_spec(AggFunc.COUNT, None, "n", DataType.INTEGER)]
        result = HashAggregate(Scan(_table()), [], specs).execute()
        assert result.num_rows == 1
        assert result.column("n").to_pylist() == [5]

    def test_all_null_group_is_null(self) -> None:
        t = Table.from_pydict({"x": [None, None]}, types={"x": DataType.INTEGER})
        specs = [_spec(AggFunc.SUM, col("x"), "s", DataType.INTEGER)]
        result = HashAggregate(Scan(t), [], specs).execute()
        assert result.column("s").to_pylist() == [None]

    def test_count_distinct(self) -> None:
        specs = [_spec(AggFunc.COUNT, col("x"), "d", DataType.INTEGER, distinct=True)]
        t = Table.from_pydict({"x": [1, 1, 2, 2, 3]}, types={"x": DataType.INTEGER})
        result = HashAggregate(Scan(t), [], specs).execute()
        assert result.column("d").to_pylist() == [3]

    def test_schema(self) -> None:
        specs = [_spec(AggFunc.COUNT, None, "n", DataType.INTEGER)]
        op = HashAggregate(Scan(_table()), [col("g").alias("g")], specs)
        assert op.schema()["n"] is DataType.INTEGER

    def test_explain(self) -> None:
        specs = [_spec(AggFunc.COUNT, None, "n", DataType.INTEGER)]
        op = HashAggregate(Scan(_table()), [col("g").alias("g")], specs)
        assert "HashAggregate" in op.explain()


class TestDistinct:
    def test_dedup_preserves_order(self) -> None:
        t = Table.from_pydict({"a": [2, 1, 2, 1, 3]})
        result = Distinct(Scan(t)).execute()
        assert result.column("a").to_pylist() == [2, 1, 3]

    def test_schema_passthrough(self) -> None:
        op = Distinct(Scan(_table()))
        assert op.schema() == {"g": DataType.TEXT, "x": DataType.INTEGER}

    def test_explain(self) -> None:
        assert "Distinct" in Distinct(Scan(_table())).explain()
