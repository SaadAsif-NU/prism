"""The type system: logical data types, inference, and promotion.

prism uses a small, explicit set of logical types. Each maps to a concrete
NumPy dtype for the values buffer of a column; nullability is tracked
separately by a validity mask (see ``column.py``), Arrow-style, so that an
integer column can hold nulls without giving up its native ``int64`` storage.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

import numpy as np


class DataType(str, Enum):
    """The logical type of a column or expression result."""

    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    TEXT = "TEXT"
    BOOLEAN = "BOOLEAN"
    # NULL is the type of an expression known only to be null (e.g. the literal
    # NULL, or a column read from data that is empty in every row). It unifies
    # with any other type.
    NULL = "NULL"

    def __repr__(self) -> str:
        return f"DataType.{self.name}"


#: Logical types backed by a fixed-width numeric NumPy dtype.
NUMERIC_TYPES = frozenset({DataType.INTEGER, DataType.FLOAT})

_NUMPY_DTYPE: dict[DataType, type] = {
    DataType.INTEGER: np.int64,
    DataType.FLOAT: np.float64,
    DataType.BOOLEAN: np.bool_,
    DataType.TEXT: np.object_,
}

_TRUE_LITERALS = frozenset({"true", "t", "yes"})
_FALSE_LITERALS = frozenset({"false", "f", "no"})


def numpy_dtype(dtype: DataType) -> type:
    """Return the NumPy dtype used to store values of a logical type.

    NULL has no concrete storage of its own, so it is materialised as an object
    array (every entry masked out).
    """
    if dtype is DataType.NULL:
        return np.object_
    return _NUMPY_DTYPE[dtype]


def is_numeric(dtype: DataType) -> bool:
    """Whether arithmetic and ordering are defined on ``dtype``."""
    return dtype in NUMERIC_TYPES


def unify(left: DataType, right: DataType) -> DataType:
    """Return the narrowest type that can hold values of both inputs.

    Used when a single result must describe values from two sources: column
    inference over many rows, the two branches of a conditional, and so on.
    NULL is the identity (it unifies to the other side); INTEGER widens to
    FLOAT; anything otherwise incompatible falls back to TEXT, which can
    represent any scalar losslessly as its string form.
    """
    if left is right:
        return left
    if left is DataType.NULL:
        return right
    if right is DataType.NULL:
        return left
    if left in NUMERIC_TYPES and right in NUMERIC_TYPES:
        return DataType.FLOAT
    return DataType.TEXT


def infer_scalar_type(token: str) -> DataType:
    """Infer the logical type of a single raw string token.

    An empty token is treated as NULL. Integers are tried before floats so
    that ``"42"`` stays an INTEGER; booleans are recognised from a small set of
    literals. Everything else is TEXT.
    """
    if token == "":
        return DataType.NULL
    lowered = token.lower()
    if lowered in _TRUE_LITERALS or lowered in _FALSE_LITERALS:
        return DataType.BOOLEAN
    if _looks_like_int(token):
        return DataType.INTEGER
    if _looks_like_float(token):
        return DataType.FLOAT
    return DataType.TEXT


def infer_column_type(tokens: Iterable[str]) -> DataType:
    """Infer a single column type covering every token in a source column.

    Folds :func:`unify` across each token's inferred type. A column that is
    empty or all-null resolves to NULL.
    """
    result = DataType.NULL
    for token in tokens:
        result = unify(result, infer_scalar_type(token))
        if result is DataType.TEXT:
            break  # TEXT absorbs everything; no need to keep scanning
    return result


def parse_value(token: str, dtype: DataType) -> object | None:
    """Parse a raw string token into a Python value of the given type.

    Returns ``None`` for nulls (empty tokens). Raises ``ValueError`` if the
    token cannot be represented as ``dtype``.
    """
    if token == "":
        return None
    if dtype is DataType.INTEGER:
        return int(token)
    if dtype is DataType.FLOAT:
        return float(token)
    if dtype is DataType.BOOLEAN:
        lowered = token.lower()
        if lowered in _TRUE_LITERALS:
            return True
        if lowered in _FALSE_LITERALS:
            return False
        raise ValueError(f"cannot parse {token!r} as BOOLEAN")
    return token


def _looks_like_int(token: str) -> bool:
    body = token[1:] if token[:1] in "+-" else token
    return body.isdigit()


def _looks_like_float(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True
