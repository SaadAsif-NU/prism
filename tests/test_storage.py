"""Tests for CSV loading, type inference on load, and the catalog."""

from __future__ import annotations

import pytest

from prism import Catalog, load_csv, load_csv_string
from prism.types import DataType


class TestCsvLoading:
    def test_basic_types_inferred(self) -> None:
        t = load_csv_string("a,b,c\n1,1.5,x\n2,2.5,y")
        assert dict(t.schema) == {
            "a": DataType.INTEGER,
            "b": DataType.FLOAT,
            "c": DataType.TEXT,
        }
        assert t.num_rows == 2

    def test_nulls_from_empty_fields(self) -> None:
        t = load_csv_string("a,b\n1,x\n,y\n3,")
        assert t.column("a").to_pylist() == [1, None, 3]
        assert t.column("b").to_pylist() == ["x", "y", None]

    def test_no_header(self) -> None:
        t = load_csv_string("1,2\n3,4", has_header=False)
        assert t.column_names == ["col0", "col1"]
        assert t.num_rows == 2

    def test_custom_delimiter(self) -> None:
        t = load_csv_string("a|b\n1|2", delimiter="|")
        assert t.column_names == ["a", "b"]

    def test_custom_null_token(self) -> None:
        t = load_csv_string("a\n1\nNA\n3", null_token="NA")
        assert t.column("a").to_pylist() == [1, None, 3]

    def test_ragged_rows_padded(self) -> None:
        t = load_csv_string("a,b,c\n1,2\n3,4,5")
        assert t.column("c").to_pylist() == [None, 5]

    def test_blank_lines_skipped(self) -> None:
        t = load_csv_string("a\n1\n\n2\n")
        assert t.column("a").to_pylist() == [1, 2]

    def test_empty_input(self) -> None:
        t = load_csv_string("")
        assert t.num_columns == 0

    def test_all_empty_column_is_text(self) -> None:
        t = load_csv_string("a,b\n1,\n2,")
        assert t.column("b").dtype is DataType.TEXT

    def test_boolean_column(self) -> None:
        t = load_csv_string("flag\ntrue\nfalse\ntrue")
        assert t.column("flag").dtype is DataType.BOOLEAN
        assert t.column("flag").to_pylist() == [True, False, True]


class TestLoadCsvFile:
    def test_load_from_disk(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "d.csv"
        path.write_text("x,y\n1,2\n3,4")
        t = load_csv(path)
        assert t.num_rows == 2


class TestCatalog:
    def test_register_and_get(self) -> None:
        cat = Catalog()
        t = load_csv_string("a\n1")
        cat.register("nums", t)
        assert cat.get("nums") is t
        assert "nums" in cat

    def test_names(self) -> None:
        cat = Catalog()
        cat.register("a", load_csv_string("x\n1"))
        cat.register("b", load_csv_string("x\n1"))
        assert set(cat.names()) == {"a", "b"}

    def test_drop(self) -> None:
        cat = Catalog()
        cat.register("a", load_csv_string("x\n1"))
        cat.drop("a")
        assert "a" not in cat
        cat.drop("a")  # dropping a missing table is a no-op

    def test_get_missing_raises(self) -> None:
        cat = Catalog()
        with pytest.raises(KeyError, match="no table"):
            cat.get("nope")

    def test_load_csv_registers(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "employees.csv"
        path.write_text("id,name\n1,Ada")
        cat = Catalog()
        cat.load_csv(path)
        assert "employees" in cat  # defaulted to the file stem

    def test_load_csv_explicit_name(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "data.csv"
        path.write_text("id\n1")
        cat = Catalog()
        cat.load_csv(path, name="staff")
        assert "staff" in cat

    def test_repr(self) -> None:
        cat = Catalog()
        cat.register("a", load_csv_string("x\n1"))
        assert "a" in repr(cat)
