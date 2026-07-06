"""Tests for the Table: schema, selection, transformation."""

from __future__ import annotations

import numpy as np
import pytest

from prism import Table
from prism.column import Column
from prism.types import DataType


class TestConstruction:
    def test_from_pydict_infers_types(self) -> None:
        t = Table.from_pydict({"a": [1, 2], "b": ["x", "y"]})
        assert dict(t.schema) == {"a": DataType.INTEGER, "b": DataType.TEXT}

    def test_from_pydict_all_null_column_is_text(self) -> None:
        t = Table.from_pydict({"a": [None, None]})
        assert t.column("a").dtype is DataType.TEXT

    def test_duplicate_names_raise(self) -> None:
        c1 = Column.from_pylist("a", [1], DataType.INTEGER)
        c2 = Column.from_pylist("a", [2], DataType.INTEGER)
        with pytest.raises(ValueError, match="duplicate"):
            Table([c1, c2])

    def test_length_mismatch_raises(self) -> None:
        c1 = Column.from_pylist("a", [1, 2], DataType.INTEGER)
        c2 = Column.from_pylist("b", [1], DataType.INTEGER)
        with pytest.raises(ValueError, match="length"):
            Table([c1, c2])

    def test_empty_table(self) -> None:
        t = Table([])
        assert t.num_rows == 0
        assert t.num_columns == 0


class TestIntrospection:
    def test_dimensions(self, people: Table) -> None:
        assert people.num_rows == 5
        assert people.num_columns == 4

    def test_column_names(self, people: Table) -> None:
        assert people.column_names == ["name", "age", "city", "salary"]

    def test_has_column(self, people: Table) -> None:
        assert people.has_column("age")
        assert not people.has_column("zzz")

    def test_column_lookup_error(self, people: Table) -> None:
        with pytest.raises(KeyError, match="no column"):
            people.column("zzz")


class TestTransformation:
    def test_select_reorders(self, people: Table) -> None:
        t = people.select(["age", "name"])
        assert t.column_names == ["age", "name"]

    def test_filter(self, people: Table) -> None:
        mask = np.array([True, False, True, False, True])
        t = people.filter(mask)
        assert t.num_rows == 3
        assert t.column("name").to_pylist() == ["Ada", "Alan", "Margaret"]

    def test_take(self, people: Table) -> None:
        t = people.take(np.array([4, 0]))
        assert t.column("name").to_pylist() == ["Margaret", "Ada"]

    def test_slice(self, people: Table) -> None:
        t = people.slice(1, 2)
        assert t.column("name").to_pylist() == ["Grace", "Alan"]

    def test_with_columns_replaces(self, people: Table) -> None:
        doubled = Column.from_pylist("age", [1, 2, 3, 4, 5], DataType.INTEGER)
        t = people.with_columns([doubled])
        assert t.num_columns == 4
        assert t.column("age").to_pylist() == [1, 2, 3, 4, 5]

    def test_with_columns_appends(self, people: Table) -> None:
        extra = Column.from_pylist("flag", [True] * 5, DataType.BOOLEAN)
        t = people.with_columns([extra])
        assert t.num_columns == 5
        assert t.has_column("flag")


class TestMaterialisation:
    def test_to_rows(self) -> None:
        t = Table.from_pydict({"a": [1, 2], "b": ["x", "y"]})
        assert t.to_rows() == [(1, "x"), (2, "y")]

    def test_to_rows_empty(self) -> None:
        assert Table([]).to_rows() == []

    def test_to_pydict_roundtrip(self) -> None:
        data = {"a": [1, None], "b": ["x", "y"]}
        t = Table.from_pydict(data)
        assert t.to_pydict() == data

    def test_repr(self, people: Table) -> None:
        assert "rows=5" in repr(people)
