"""Tests for the logical type system: inference, unification, parsing."""

from __future__ import annotations

import pytest

from prism.types import (
    DataType,
    infer_column_type,
    infer_scalar_type,
    is_numeric,
    numpy_dtype,
    parse_value,
    unify,
)


class TestScalarInference:
    def test_empty_is_null(self) -> None:
        assert infer_scalar_type("") is DataType.NULL

    def test_integers(self) -> None:
        assert infer_scalar_type("42") is DataType.INTEGER
        assert infer_scalar_type("-7") is DataType.INTEGER
        assert infer_scalar_type("+3") is DataType.INTEGER

    def test_floats(self) -> None:
        assert infer_scalar_type("3.14") is DataType.FLOAT
        assert infer_scalar_type("-0.5") is DataType.FLOAT
        assert infer_scalar_type("1e9") is DataType.FLOAT

    def test_booleans(self) -> None:
        assert infer_scalar_type("true") is DataType.BOOLEAN
        assert infer_scalar_type("FALSE") is DataType.BOOLEAN
        assert infer_scalar_type("yes") is DataType.BOOLEAN

    def test_text(self) -> None:
        assert infer_scalar_type("hello") is DataType.TEXT
        assert infer_scalar_type("12abc") is DataType.TEXT


class TestColumnInference:
    def test_all_integers(self) -> None:
        assert infer_column_type(["1", "2", "3"]) is DataType.INTEGER

    def test_mixed_int_float_widens(self) -> None:
        assert infer_column_type(["1", "2.5", "3"]) is DataType.FLOAT

    def test_nulls_ignored(self) -> None:
        assert infer_column_type(["1", "", "3"]) is DataType.INTEGER

    def test_all_null(self) -> None:
        assert infer_column_type(["", "", ""]) is DataType.NULL

    def test_text_absorbs(self) -> None:
        assert infer_column_type(["1", "x", "2"]) is DataType.TEXT

    def test_empty_iterable(self) -> None:
        assert infer_column_type([]) is DataType.NULL


class TestUnify:
    def test_identity(self) -> None:
        assert unify(DataType.INTEGER, DataType.INTEGER) is DataType.INTEGER

    def test_null_is_identity(self) -> None:
        assert unify(DataType.NULL, DataType.TEXT) is DataType.TEXT
        assert unify(DataType.FLOAT, DataType.NULL) is DataType.FLOAT

    def test_numeric_widening(self) -> None:
        assert unify(DataType.INTEGER, DataType.FLOAT) is DataType.FLOAT

    def test_incompatible_falls_back_to_text(self) -> None:
        assert unify(DataType.INTEGER, DataType.BOOLEAN) is DataType.TEXT


class TestParsing:
    def test_parse_null(self) -> None:
        assert parse_value("", DataType.INTEGER) is None

    def test_parse_int(self) -> None:
        assert parse_value("42", DataType.INTEGER) == 42

    def test_parse_float(self) -> None:
        assert parse_value("3.5", DataType.FLOAT) == 3.5

    def test_parse_bool(self) -> None:
        assert parse_value("true", DataType.BOOLEAN) is True
        assert parse_value("no", DataType.BOOLEAN) is False

    def test_parse_bad_bool(self) -> None:
        with pytest.raises(ValueError):
            parse_value("maybe", DataType.BOOLEAN)

    def test_parse_text(self) -> None:
        assert parse_value("hello", DataType.TEXT) == "hello"


class TestMisc:
    def test_is_numeric(self) -> None:
        assert is_numeric(DataType.INTEGER)
        assert is_numeric(DataType.FLOAT)
        assert not is_numeric(DataType.TEXT)
        assert not is_numeric(DataType.BOOLEAN)

    def test_numpy_dtype_null_is_object(self) -> None:
        import numpy as np

        assert numpy_dtype(DataType.NULL) is np.object_

    def test_repr(self) -> None:
        assert repr(DataType.INTEGER) == "DataType.INTEGER"
