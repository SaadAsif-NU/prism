"""Tests for the physical operators and the fluent Relation API."""

from __future__ import annotations

import pytest

from prism import Relation, Table, col, lit
from prism.exec.operators import Filter, Limit, Project, Scan, Sort, SortKey
from prism.types import DataType


class TestScan:
    def test_scan_returns_table(self, people: Table) -> None:
        scan = Scan(people, "people")
        assert scan.execute() is people
        assert scan.schema()["age"] is DataType.INTEGER

    def test_describe(self, people: Table) -> None:
        assert "Scan" in Scan(people, "people").explain()


class TestFilter:
    def test_filter_keeps_true_rows(self, people: Table) -> None:
        op = Filter(Scan(people), col("age") > 40)
        result = op.execute()
        assert result.column("name").to_pylist() == ["Grace", "Alan"]

    def test_null_predicate_drops_row(self, people: Table) -> None:
        # Katherine has a null age, so age > 0 is NULL and she is excluded.
        op = Filter(Scan(people), col("age") > 0)
        assert op.execute().num_rows == 4

    def test_non_boolean_predicate_raises(self, people: Table) -> None:
        op = Filter(Scan(people), col("age") + 1)
        with pytest.raises(TypeError, match="BOOLEAN"):
            op.execute()


class TestProject:
    def test_project_subset(self, people: Table) -> None:
        op = Project(Scan(people), [col("name")])
        assert op.execute().column_names == ["name"]

    def test_project_computed(self, people: Table) -> None:
        op = Project(Scan(people), [(col("salary") / 1000).alias("k")])
        result = op.execute()
        assert result.column("k").to_pylist()[0] == 145.0

    def test_project_schema(self, people: Table) -> None:
        op = Project(Scan(people), [col("name"), (col("age") + 1).alias("next_age")])
        schema = op.schema()
        assert schema["next_age"] is DataType.INTEGER


class TestSort:
    def test_sort_ascending(self, people: Table) -> None:
        op = Sort(Scan(people), [SortKey(col("salary"))])
        assert op.execute().column("name").to_pylist()[0] == "Alan"

    def test_sort_descending(self, people: Table) -> None:
        op = Sort(Scan(people), [SortKey(col("salary"), ascending=False)])
        assert op.execute().column("name").to_pylist()[0] == "Grace"

    def test_sort_nulls_last_by_default(self, people: Table) -> None:
        # age has a null (Katherine); ascending should place her last.
        op = Sort(Scan(people), [SortKey(col("age"))])
        assert op.execute().column("name").to_pylist()[-1] == "Katherine"

    def test_sort_nulls_first(self, people: Table) -> None:
        op = Sort(Scan(people), [SortKey(col("age"), nulls_first=True)])
        assert op.execute().column("name").to_pylist()[0] == "Katherine"

    def test_multi_key_sort(self) -> None:
        t = Table.from_pydict({"grp": ["b", "a", "b", "a"], "n": [2, 1, 1, 2]})
        op = Sort(Scan(t), [SortKey(col("grp")), SortKey(col("n"))])
        rows = op.execute().to_rows()
        assert rows == [("a", 1), ("a", 2), ("b", 1), ("b", 2)]

    def test_sort_text(self) -> None:
        t = Table.from_pydict({"w": ["pear", "apple", "kiwi"]})
        op = Sort(Scan(t), [SortKey(col("w"))])
        assert op.execute().column("w").to_pylist() == ["apple", "kiwi", "pear"]

    def test_empty_keys_raise(self, people: Table) -> None:
        with pytest.raises(ValueError, match="at least one key"):
            Sort(Scan(people), [])

    def test_single_row_passthrough(self) -> None:
        t = Table.from_pydict({"a": [1]})
        op = Sort(Scan(t), [SortKey(col("a"))])
        assert op.execute().num_rows == 1


class TestLimit:
    def test_limit(self, people: Table) -> None:
        op = Limit(Scan(people), 2)
        assert op.execute().num_rows == 2

    def test_offset(self, people: Table) -> None:
        op = Limit(Scan(people), 2, offset=1)
        assert op.execute().column("name").to_pylist() == ["Grace", "Alan"]

    def test_negative_limit_raises(self, people: Table) -> None:
        with pytest.raises(ValueError):
            Limit(Scan(people), -1)

    def test_negative_offset_raises(self, people: Table) -> None:
        with pytest.raises(ValueError):
            Limit(Scan(people), 1, offset=-1)


class TestRelationApi:
    def test_end_to_end_pipeline(self, people: Table) -> None:
        result = (
            Relation.from_table(people, "people")
            .filter(col("age") > 30)
            .select("name", col("age"))
            .sort(col("age"), ascending=False)
            .limit(2)
            .collect()
        )
        assert result.column("name").to_pylist() == ["Grace", "Alan"]

    def test_select_accepts_strings_and_exprs(self, people: Table) -> None:
        result = Relation.from_table(people).select("name", (col("age") + 1).alias("a1"))
        assert result.schema()["a1"] is DataType.INTEGER

    def test_sort_with_sortkey(self, people: Table) -> None:
        result = Relation.from_table(people).sort(SortKey(col("salary"), ascending=False)).collect()
        assert result.column("name").to_pylist()[0] == "Grace"

    def test_explain_renders_tree(self, people: Table) -> None:
        rel = Relation.from_table(people).filter(col("age") > 1).limit(3)
        text = rel.explain()
        assert "Limit" in text and "Filter" in text and "Scan" in text

    def test_schema_without_execution(self, people: Table) -> None:
        rel = Relation.from_table(people).select("name")
        assert rel.schema() == {"name": DataType.TEXT}

    def test_repr(self, people: Table) -> None:
        rel = Relation.from_table(people).limit(1)
        assert "Relation" in repr(rel)

    def test_literal_projection(self, people: Table) -> None:
        result = Relation.from_table(people).select(lit(1).alias("one")).collect()
        assert result.column("one").to_pylist() == [1, 1, 1, 1, 1]
