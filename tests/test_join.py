"""Unit tests for the join operators."""

from __future__ import annotations

from prism import Table, col
from prism.exec.join import HashJoin, NestedLoopJoin
from prism.exec.operators import Scan
from prism.types import DataType


def _left() -> Table:
    return Table.from_pydict(
        {"id": [1, 2, 3], "name": ["a", "b", "c"]},
        types={"id": DataType.INTEGER, "name": DataType.TEXT},
    )


def _right() -> Table:
    return Table.from_pydict(
        {"rid": [1, 2, 2, 4], "tag": ["x", "y", "z", "w"]},
        types={"rid": DataType.INTEGER, "tag": DataType.TEXT},
    )


class TestHashJoin:
    def test_inner_join(self) -> None:
        op = HashJoin(Scan(_left()), Scan(_right()), [col("id")], [col("rid")], "INNER")
        result = op.execute()
        pairs = sorted(
            zip(result.column("name").to_pylist(), result.column("tag").to_pylist(), strict=True)
        )
        assert pairs == [("a", "x"), ("b", "y"), ("b", "z")]

    def test_left_join_keeps_unmatched(self) -> None:
        op = HashJoin(Scan(_left()), Scan(_right()), [col("id")], [col("rid")], "LEFT")
        result = op.execute()
        by_name = dict(
            zip(result.column("name").to_pylist(), result.column("tag").to_pylist(), strict=True)
        )
        assert by_name["c"] is None  # id 3 has no match

    def test_null_keys_never_match(self) -> None:
        left = Table.from_pydict(
            {"id": [None, 1], "v": ["p", "q"]}, types={"id": DataType.INTEGER, "v": DataType.TEXT}
        )
        right = Table.from_pydict(
            {"rid": [None, 1], "t": ["x", "y"]}, types={"rid": DataType.INTEGER, "t": DataType.TEXT}
        )
        op = HashJoin(Scan(left), Scan(right), [col("id")], [col("rid")], "INNER")
        result = op.execute()
        # Only 1 = 1 matches; the two NULL keys do not.
        assert result.num_rows == 1

    def test_residual_predicate(self) -> None:
        # Equi-join on id=rid, plus a residual that keeps only tag = 'z'.
        op = HashJoin(
            Scan(_left()),
            Scan(_right()),
            [col("id")],
            [col("rid")],
            "INNER",
            residual=col("tag") == "z",
        )
        result = op.execute()
        assert result.column("tag").to_pylist() == ["z"]

    def test_empty_right(self) -> None:
        empty = Table.from_pydict(
            {"rid": [], "tag": []}, types={"rid": DataType.INTEGER, "tag": DataType.TEXT}
        )
        op = HashJoin(Scan(_left()), Scan(empty), [col("id")], [col("rid")], "LEFT")
        result = op.execute()
        assert result.num_rows == 3
        assert result.column("tag").to_pylist() == [None, None, None]

    def test_schema_merges_sides(self) -> None:
        op = HashJoin(Scan(_left()), Scan(_right()), [col("id")], [col("rid")], "INNER")
        assert set(op.schema()) == {"id", "name", "rid", "tag"}

    def test_explain(self) -> None:
        op = HashJoin(Scan(_left()), Scan(_right()), [col("id")], [col("rid")], "INNER")
        assert "HashJoin" in op.explain()


class TestNestedLoopJoin:
    def test_cross_join(self) -> None:
        op = NestedLoopJoin(Scan(_left()), Scan(_right()), None, "INNER")
        assert op.execute().num_rows == 12  # 3 x 4

    def test_predicate_join(self) -> None:
        op = NestedLoopJoin(Scan(_left()), Scan(_right()), col("id") == col("rid"), "INNER")
        assert op.execute().num_rows == 3  # (1,1), (2,2), (2,2)

    def test_left_join(self) -> None:
        op = NestedLoopJoin(Scan(_left()), Scan(_right()), col("id") == col("rid"), "LEFT")
        result = op.execute()
        assert result.num_rows == 4  # 3 matches + unmatched id 3

    def test_empty_right_left_join(self) -> None:
        empty = Table.from_pydict(
            {"rid": [], "tag": []}, types={"rid": DataType.INTEGER, "tag": DataType.TEXT}
        )
        op = NestedLoopJoin(Scan(_left()), Scan(empty), col("id") == col("rid"), "LEFT")
        assert op.execute().num_rows == 3

    def test_explain(self) -> None:
        op = NestedLoopJoin(Scan(_left()), Scan(_right()), None, "INNER")
        assert "NestedLoopJoin" in op.explain()
