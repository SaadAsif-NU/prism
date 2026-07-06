"""Tests for the Column: nullable typed storage and selection."""

from __future__ import annotations

import numpy as np
import pytest

from prism.column import Column
from prism.types import DataType


class TestConstruction:
    def test_from_pylist_no_nulls(self) -> None:
        col = Column.from_pylist("a", [1, 2, 3], DataType.INTEGER)
        assert len(col) == 3
        assert col.null_count == 0
        assert not col.has_nulls
        assert col.to_pylist() == [1, 2, 3]

    def test_from_pylist_with_nulls(self) -> None:
        col = Column.from_pylist("a", [1, None, 3], DataType.INTEGER)
        assert col.null_count == 1
        assert col.has_nulls
        assert col.to_pylist() == [1, None, 3]
        # The null slot keeps the buffer well-formed with a typed placeholder.
        assert col.values.dtype == np.int64

    def test_from_pylist_empty(self) -> None:
        col = Column.from_pylist("a", [], DataType.FLOAT)
        assert len(col) == 0
        assert col.values.dtype == np.float64

    def test_from_values_defaults_all_valid(self) -> None:
        col = Column.from_values("a", DataType.INTEGER, np.array([1, 2, 3]))
        assert col.null_count == 0

    def test_text_nulls(self) -> None:
        col = Column.from_pylist("a", ["x", None, "z"], DataType.TEXT)
        assert col.to_pylist() == ["x", None, "z"]

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            Column("a", DataType.INTEGER, np.array([1, 2]), np.array([True]))

    def test_non_bool_validity_raises(self) -> None:
        with pytest.raises(ValueError, match="boolean"):
            Column("a", DataType.INTEGER, np.array([1]), np.array([1]))


class TestSelection:
    def test_take(self) -> None:
        col = Column.from_pylist("a", [10, 20, 30], DataType.INTEGER)
        taken = col.take(np.array([2, 0]))
        assert taken.to_pylist() == [30, 10]

    def test_take_preserves_nulls(self) -> None:
        col = Column.from_pylist("a", [10, None, 30], DataType.INTEGER)
        taken = col.take(np.array([1, 2]))
        assert taken.to_pylist() == [None, 30]

    def test_filter(self) -> None:
        col = Column.from_pylist("a", [1, 2, 3, 4], DataType.INTEGER)
        kept = col.filter(np.array([True, False, True, False]))
        assert kept.to_pylist() == [1, 3]

    def test_slice(self) -> None:
        col = Column.from_pylist("a", [1, 2, 3, 4, 5], DataType.INTEGER)
        assert col.slice(1, 2).to_pylist() == [2, 3]
        assert col.slice(3).to_pylist() == [4, 5]


class TestMisc:
    def test_rename_shares_buffers(self) -> None:
        col = Column.from_pylist("a", [1, 2], DataType.INTEGER)
        renamed = col.rename("b")
        assert renamed.name == "b"
        assert renamed.values is col.values

    def test_repr(self) -> None:
        col = Column.from_pylist("a", [1, None], DataType.INTEGER)
        text = repr(col)
        assert "Column" in text and "nulls=1" in text
