"""Render a result :class:`~prism.table.Table` as a bordered text grid.

Used by the CLI shell and anywhere a table needs a readable, aligned dump.
Numbers are right-aligned, NULLs render as ``NULL``, and overly wide cells are
truncated with an ellipsis so a single wide row cannot wreck the layout.
"""

from __future__ import annotations

from prism.table import Table
from prism.types import is_numeric

_MAX_WIDTH = 40


def render_table(table: Table, max_rows: int | None = 50) -> str:
    """Return a box-drawing rendering of ``table``.

    At most ``max_rows`` data rows are shown; if more exist, a summary line
    notes how many were elided. Pass ``None`` to show every row.
    """
    headers = table.column_names
    types = [t for _, t in table.schema]
    if not headers:
        return "(no columns)"

    rows = table.to_rows()
    shown = rows if max_rows is None else rows[:max_rows]
    cells = [[_render_cell(v) for v in row] for row in shown]

    widths = [len(h) for h in headers]
    for i, t in enumerate(types):
        widths[i] = max(widths[i], len(t.value))
    for row in cells:
        for i, text in enumerate(row):
            widths[i] = min(max(widths[i], len(text)), _MAX_WIDTH)

    aligns = [is_numeric(t) for t in types]
    lines = [
        _border("┌", "┬", "┐", widths),
        _row(headers, widths, [False] * len(widths)),
        _row([t.value for t in types], widths, [False] * len(widths)),
        _border("├", "┼", "┤", widths),
    ]
    lines.extend(_row(row, widths, aligns) for row in cells)
    lines.append(_border("└", "┴", "┘", widths))

    summary = _summary(len(rows), len(shown))
    lines.append(summary)
    return "\n".join(lines)


def _render_cell(value: object | None) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = f"{value:.6g}"
        return text
    return str(value)


def _truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def _row(values: list[str], widths: list[int], right_align: list[bool]) -> str:
    cells = []
    for text, width, right in zip(values, widths, right_align, strict=True):
        clipped = _truncate(text, width)
        cell = clipped.rjust(width) if right else clipped.ljust(width)
        cells.append(f" {cell} ")
    return "│" + "│".join(cells) + "│"


def _border(left: str, mid: str, right: str, widths: list[int]) -> str:
    segments = ["─" * (w + 2) for w in widths]
    return left + mid.join(segments) + right


def _summary(total: int, shown: int) -> str:
    noun = "row" if total == 1 else "rows"
    if shown < total:
        return f"{total} {noun} ({shown} shown)"
    return f"{total} {noun} in set"


def render_schema(name: str, table: Table) -> str:
    """Return a compact description of a table's columns and types."""
    lines = [f"{name} ({table.num_rows} rows)"]
    width = max((len(n) for n in table.column_names), default=0)
    for col_name, dtype in table.schema:
        lines.append(f"  {col_name.ljust(width)}  {dtype.value}")
    return "\n".join(lines)
