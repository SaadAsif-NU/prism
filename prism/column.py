"""Columnar storage: a typed, nullable array of values.

A :class:`Column` is the unit of storage in prism. Values live in a single
contiguous NumPy array of the column's native dtype; nullability is carried by
a separate boolean *validity mask* (``True`` means present, ``False`` means
NULL). This is the same split Apache Arrow and pandas' nullable dtypes use, and
it is what makes an integer column able to hold NULLs without decaying to
``float`` or ``object``. Storing values column-by-column (rather than
row-by-row) is what lets the execution engine operate on whole arrays at once.
"""

from __future__ import annotations

import numpy as np

from prism.types import DataType, numpy_dtype

#: Value written into the values buffer at masked-out (null) positions. Never
#: read back through the public API, but keeps the typed buffer well-formed.
_PLACEHOLDER: dict[DataType, object] = {
    DataType.INTEGER: 0,
    DataType.FLOAT: np.nan,
    DataType.BOOLEAN: False,
    DataType.TEXT: "",
    DataType.NULL: None,
}


class Column:
    """A named, typed, nullable sequence of values stored column-major."""

    __slots__ = ("name", "dtype", "values", "validity")

    def __init__(
        self,
        name: str,
        dtype: DataType,
        values: np.ndarray,
        validity: np.ndarray,
    ) -> None:
        if len(values) != len(validity):
            raise ValueError(
                f"values ({len(values)}) and validity ({len(validity)}) length mismatch"
            )
        if validity.dtype != np.bool_:
            raise ValueError("validity mask must be a boolean array")
        self.name = name
        self.dtype = dtype
        self.values = values
        self.validity = validity

    # -- construction ---------------------------------------------------

    @classmethod
    def from_pylist(
        cls,
        name: str,
        data: list[object | None],
        dtype: DataType,
    ) -> Column:
        """Build a column from Python values, using ``None`` to mark nulls."""
        validity = np.array([v is not None for v in data], dtype=np.bool_)
        if not data:
            # np.array([]) defaults to float64; force the declared storage type.
            values: np.ndarray = np.empty(0, dtype=numpy_dtype(dtype))
        else:
            placeholder = _PLACEHOLDER[dtype]
            filled = [placeholder if v is None else v for v in data]
            values = np.array(filled, dtype=numpy_dtype(dtype))
        return cls(name, dtype, values, validity)

    @classmethod
    def from_values(
        cls,
        name: str,
        dtype: DataType,
        values: np.ndarray,
        validity: np.ndarray | None = None,
    ) -> Column:
        """Wrap an existing values buffer, defaulting to all-valid."""
        if validity is None:
            validity = np.ones(len(values), dtype=np.bool_)
        return cls(name, dtype, values, validity)

    # -- introspection --------------------------------------------------

    def __len__(self) -> int:
        return len(self.values)

    @property
    def null_count(self) -> int:
        """Number of NULLs in the column."""
        return int((~self.validity).sum())

    @property
    def has_nulls(self) -> bool:
        return not self.validity.all()

    def rename(self, name: str) -> Column:
        """Return a view of this column under a new name (buffers shared)."""
        return Column(name, self.dtype, self.values, self.validity)

    # -- selection ------------------------------------------------------

    def take(self, indices: np.ndarray) -> Column:
        """Gather rows at ``indices`` (used by sort, join, and reordering)."""
        return Column(self.name, self.dtype, self.values[indices], self.validity[indices])

    def filter(self, mask: np.ndarray) -> Column:
        """Keep only rows where the boolean ``mask`` is True."""
        return Column(self.name, self.dtype, self.values[mask], self.validity[mask])

    def slice(self, start: int, length: int | None = None) -> Column:
        """Return a contiguous row range (used by LIMIT / OFFSET)."""
        stop = None if length is None else start + length
        return Column(self.name, self.dtype, self.values[start:stop], self.validity[start:stop])

    # -- materialisation ------------------------------------------------

    def to_pylist(self) -> list[object | None]:
        """Return the column as Python values, with ``None`` for NULLs."""
        out: list[object | None] = []
        for value, valid in zip(self.values.tolist(), self.validity.tolist(), strict=True):
            out.append(value if valid else None)
        return out

    def __repr__(self) -> str:
        return (
            f"Column(name={self.name!r}, dtype={self.dtype.value}, "
            f"len={len(self)}, nulls={self.null_count})"
        )
