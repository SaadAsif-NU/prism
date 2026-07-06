"""Load CSV text into a columnar :class:`~prism.table.Table`.

The loader reads the file row by row, transposes it into columns, infers a
logical type per column from its values, and parses each column into a typed,
nullable buffer. Type inference follows the rules in :mod:`prism.types`:
integers stay integers, a column mixing integers and floats widens to FLOAT,
and anything non-numeric falls back to TEXT.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from prism.column import Column
from prism.table import Table
from prism.types import DataType, infer_column_type, parse_value


def load_csv(
    path: str | Path,
    *,
    delimiter: str = ",",
    null_token: str = "",
    has_header: bool = True,
) -> Table:
    """Load a CSV file from disk into a table."""
    text = Path(path).read_text(encoding="utf-8")
    return load_csv_string(
        text,
        delimiter=delimiter,
        null_token=null_token,
        has_header=has_header,
    )


def load_csv_string(
    text: str,
    *,
    delimiter: str = ",",
    null_token: str = "",
    has_header: bool = True,
) -> Table:
    """Load CSV content from a string (the workhorse; handy for tests)."""
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [row for row in reader if row]  # skip fully blank lines
    if not rows:
        return Table([])

    if has_header:
        header, body = rows[0], rows[1:]
    else:
        width = len(rows[0])
        header = [f"col{i}" for i in range(width)]
        body = rows

    width = len(header)
    # Transpose rows into columns of raw string tokens, normalising nulls and
    # padding short rows so ragged input does not crash the loader.
    raw_columns: list[list[str]] = [[] for _ in range(width)]
    for row in body:
        for i in range(width):
            token = row[i] if i < len(row) else ""
            raw_columns[i].append("" if token == null_token else token)

    columns: list[Column] = []
    for name, tokens in zip(header, raw_columns, strict=True):
        dtype = infer_column_type(tokens)
        if dtype is DataType.NULL:
            dtype = DataType.TEXT  # an all-empty column is typeless; call it TEXT
        parsed = [parse_value(tok, dtype) for tok in tokens]
        columns.append(Column.from_pylist(name, parsed, dtype))
    return Table(columns)
