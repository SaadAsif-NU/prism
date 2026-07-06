"""Physical operators: a pull-based tree that produces the query result.

Each operator exposes ``execute() -> Table`` and pulls its input from a child
operator, so a query is a tree of these nodes with a :class:`Scan` at every
leaf. Execution is vectorised: an operator transforms whole columns at once
rather than looping over rows. ``schema()`` returns the output types without
touching data, which lets the planner and optimizer reason about a plan before
it runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from prism.column import Column
from prism.expr import Expression, Schema
from prism.table import Table
from prism.types import DataType, is_numeric


class Operator(ABC):
    """A node in the physical execution tree."""

    @abstractmethod
    def execute(self) -> Table:
        """Run this operator (and its children) and return the result table."""

    @abstractmethod
    def schema(self) -> Schema:
        """Return the output column types without executing."""

    @property
    def children(self) -> Sequence[Operator]:
        """Child operators, for plan traversal and EXPLAIN."""
        return ()

    def explain(self, indent: int = 0) -> str:
        """Render this subtree as an indented textual plan."""
        pad = "  " * indent
        lines = [f"{pad}{self._describe()}"]
        for child in self.children:
            lines.append(child.explain(indent + 1))
        return "\n".join(lines)

    def _describe(self) -> str:
        return type(self).__name__


class Scan(Operator):
    """Leaf operator: reads rows from a materialised base table."""

    def __init__(self, table: Table, name: str = "?") -> None:
        self.table = table
        self.name = name

    def execute(self) -> Table:
        return self.table

    def schema(self) -> Schema:
        return dict(self.table.schema)

    def _describe(self) -> str:
        return f"Scan({self.name}, rows={self.table.num_rows})"


class Filter(Operator):
    """Keeps rows for which ``predicate`` evaluates to TRUE.

    A NULL predicate result is treated as "not true", so those rows are
    dropped, matching SQL ``WHERE`` semantics.
    """

    def __init__(self, child: Operator, predicate: Expression) -> None:
        self.child = child
        self.predicate = predicate

    def execute(self) -> Table:
        table = self.child.execute()
        result = self.predicate.evaluate(table)
        if result.dtype is not DataType.BOOLEAN:
            raise TypeError(f"WHERE predicate must be BOOLEAN, got {result.dtype}")
        mask = result.values.astype(np.bool_) & result.validity
        return table.filter(mask)

    def schema(self) -> Schema:
        return self.child.schema()

    @property
    def children(self) -> Sequence[Operator]:
        return (self.child,)

    def _describe(self) -> str:
        return f"Filter({self.predicate!r})"


class Project(Operator):
    """Computes a new set of output columns from ``projections``."""

    def __init__(self, child: Operator, projections: Sequence[Expression]) -> None:
        self.child = child
        self.projections = list(projections)

    def execute(self) -> Table:
        table = self.child.execute()
        columns = [expr.evaluate(table) for expr in self.projections]
        return Table(columns)

    def schema(self) -> Schema:
        child_schema = self.child.schema()
        return {expr.output_name(): expr.resolve_type(child_schema) for expr in self.projections}

    @property
    def children(self) -> Sequence[Operator]:
        return (self.child,)

    def _describe(self) -> str:
        cols = ", ".join(e.output_name() for e in self.projections)
        return f"Project([{cols}])"


@dataclass(frozen=True)
class SortKey:
    """One ORDER BY term: an expression, a direction, and null placement.

    ``nulls_first`` defaults to SQL's convention (NULLs sort as if larger than
    any value: last under ASC, first under DESC) when left as ``None``.
    """

    expression: Expression
    ascending: bool = True
    nulls_first: bool | None = None

    def resolved_nulls_first(self) -> bool:
        if self.nulls_first is not None:
            return self.nulls_first
        return not self.ascending


class Sort(Operator):
    """Orders rows by one or more sort keys (stable, multi-key)."""

    def __init__(self, child: Operator, keys: Sequence[SortKey]) -> None:
        if not keys:
            raise ValueError("Sort requires at least one key")
        self.child = child
        self.keys = list(keys)

    def execute(self) -> Table:
        table = self.child.execute()
        if table.num_rows <= 1:
            return table
        # np.lexsort takes keys least-significant first, so reverse ORDER BY.
        lex_keys = [
            _sort_key_array(
                key.expression.evaluate(table), key.ascending, key.resolved_nulls_first()
            )
            for key in reversed(self.keys)
        ]
        order = np.lexsort(lex_keys)
        return table.take(order)

    def schema(self) -> Schema:
        return self.child.schema()

    @property
    def children(self) -> Sequence[Operator]:
        return (self.child,)

    def _describe(self) -> str:
        terms = ", ".join(f"{k.expression!r} {'ASC' if k.ascending else 'DESC'}" for k in self.keys)
        return f"Sort([{terms}])"


class Limit(Operator):
    """Returns at most ``limit`` rows after skipping ``offset`` rows."""

    def __init__(self, child: Operator, limit: int | None, offset: int = 0) -> None:
        if limit is not None and limit < 0:
            raise ValueError("LIMIT must be non-negative")
        if offset < 0:
            raise ValueError("OFFSET must be non-negative")
        self.child = child
        self.limit = limit
        self.offset = offset

    def execute(self) -> Table:
        table = self.child.execute()
        return table.slice(self.offset, self.limit)

    def schema(self) -> Schema:
        return self.child.schema()

    @property
    def children(self) -> Sequence[Operator]:
        return (self.child,)

    def _describe(self) -> str:
        return f"Limit(limit={self.limit}, offset={self.offset})"


# ----------------------------------------------------------------------
# sort key encoding
# ----------------------------------------------------------------------


def _sort_key_array(column: Column, ascending: bool, nulls_first: bool) -> np.ndarray:
    """Encode a column as a float64 key that lexsort orders correctly.

    Direction is folded into the sign of the key and NULLs are mapped to an
    infinite sentinel, so every key can be sorted ascending by ``np.lexsort``.
    TEXT is rank-encoded to integer codes that preserve lexicographic order.
    """
    if is_numeric(column.dtype) or column.dtype is DataType.BOOLEAN:
        base = column.values.astype(np.float64)
    else:
        # Factorise strings into order-preserving integer codes.
        _, codes = np.unique(column.values, return_inverse=True)
        base = codes.astype(np.float64)
    sign = 1.0 if ascending else -1.0
    key = sign * base
    sentinel = np.inf if not nulls_first else -np.inf
    return np.where(column.validity, key, sentinel)
