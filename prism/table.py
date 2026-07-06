"""A table: an ordered set of equal-length named columns, plus its schema.

Tables are immutable value objects. Every transformation (select, filter,
take, slice) returns a new table that shares the underlying column buffers
where possible, so plans compose without defensive copying.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from prism.column import Column
from prism.types import DataType


class Table:
    """An ordered collection of named columns of equal length."""

    __slots__ = ("columns", "_index")

    def __init__(self, columns: Sequence[Column]) -> None:
        names = [c.name for c in columns]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate column names: {names}")
        if columns:
            length = len(columns[0])
            for c in columns:
                if len(c) != length:
                    raise ValueError(f"column {c.name!r} has length {len(c)}, expected {length}")
        self.columns: tuple[Column, ...] = tuple(columns)
        self._index = {c.name: i for i, c in enumerate(self.columns)}

    # -- construction ---------------------------------------------------

    @classmethod
    def from_pydict(
        cls,
        data: dict[str, list[object | None]],
        types: dict[str, DataType] | None = None,
    ) -> Table:
        """Build a table from a dict of column name to Python values.

        Types may be supplied per column; any omitted are inferred from the
        values via the same rules the CSV loader uses.
        """
        from prism.types import infer_column_type

        types = types or {}
        cols: list[Column] = []
        for name, values in data.items():
            dtype = types.get(name)
            if dtype is None:
                tokens = ["" if v is None else str(v) for v in values]
                dtype = infer_column_type(tokens)
                if dtype is DataType.NULL:
                    dtype = DataType.TEXT
            cols.append(Column.from_pylist(name, values, dtype))
        return cls(cols)

    # -- introspection --------------------------------------------------

    @property
    def num_rows(self) -> int:
        return len(self.columns[0]) if self.columns else 0

    @property
    def num_columns(self) -> int:
        return len(self.columns)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def schema(self) -> list[tuple[str, DataType]]:
        return [(c.name, c.dtype) for c in self.columns]

    def has_column(self, name: str) -> bool:
        return name in self._index

    def column(self, name: str) -> Column:
        try:
            return self.columns[self._index[name]]
        except KeyError:
            raise KeyError(f"no column named {name!r} (have {self.column_names})") from None

    def __len__(self) -> int:
        return self.num_rows

    # -- transformation -------------------------------------------------

    def select(self, names: Sequence[str]) -> Table:
        """Return a table with the named columns, in the order given."""
        return Table([self.column(n) for n in names])

    def with_columns(self, columns: Sequence[Column]) -> Table:
        """Return a table with ``columns`` appended or replaced by name."""
        merged = {c.name: c for c in self.columns}
        for c in columns:
            merged[c.name] = c
        return Table(list(merged.values()))

    def filter(self, mask: np.ndarray) -> Table:
        """Keep rows where the boolean ``mask`` is True."""
        return Table([c.filter(mask) for c in self.columns])

    def take(self, indices: np.ndarray) -> Table:
        """Gather rows at ``indices`` (reordering or repeating)."""
        return Table([c.take(indices) for c in self.columns])

    def slice(self, start: int, length: int | None = None) -> Table:
        """Return a contiguous range of rows."""
        return Table([c.slice(start, length) for c in self.columns])

    # -- materialisation ------------------------------------------------

    def to_rows(self) -> list[tuple[object | None, ...]]:
        """Return the table as a list of row tuples (values, ``None`` = NULL)."""
        cols = [c.to_pylist() for c in self.columns]
        return list(zip(*cols, strict=True)) if cols else []

    def to_pydict(self) -> dict[str, list[object | None]]:
        """Return the table as a dict of column name to Python values."""
        return {c.name: c.to_pylist() for c in self.columns}

    def __repr__(self) -> str:
        schema = ", ".join(f"{n}:{t.value}" for n, t in self.schema)
        return f"Table(rows={self.num_rows}, columns=[{schema}])"
