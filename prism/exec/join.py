"""Join operators: hash join for equi-joins, nested loop as the fallback.

Both support ``INNER`` and ``LEFT`` joins. The planner extracts equality
conjuncts from the ``ON`` clause to drive :class:`HashJoin`; any remaining
non-equi condition rides along as a ``residual`` predicate applied to candidate
pairs. When there are no equalities to hash on, the planner falls back to
:class:`NestedLoopJoin`, which evaluates the full predicate over every pair.

Column names are assumed globally unique across the two inputs; the planner
guarantees this by qualifying overlapping names before the join is built.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from prism.column import Column
from prism.exec.operators import Operator
from prism.expr import Expression, Schema
from prism.table import Table


def _combine(left: Table, right: Table, left_idx: np.ndarray, right_idx: np.ndarray) -> Table:
    """Assemble output rows from paired row indices (right idx -1 means NULL)."""
    matched = right_idx >= 0
    columns: list[Column] = []

    if left.num_columns:
        left_taken = left.take(left_idx) if len(left_idx) else left.slice(0, 0)
        columns.extend(left_taken.columns)

    for c in right.columns:
        if right.num_rows == 0:
            values = np.empty(len(right_idx), dtype=c.values.dtype)
            validity = np.zeros(len(right_idx), dtype=np.bool_)
        else:
            clamped = np.where(matched, right_idx, 0)
            values = c.values[clamped]
            validity = c.validity[clamped] & matched
        columns.append(Column(c.name, c.dtype, values, validity))
    return Table(columns)


def _finalize(
    left: Table,
    right: Table,
    cand_left: list[int],
    cand_right: list[int],
    how: str,
    residual: Expression | None,
) -> Table:
    """Filter candidate pairs by the residual and add unmatched LEFT rows."""
    if residual is not None and cand_left:
        combined = _combine(left, right, np.array(cand_left), np.array(cand_right))
        result = residual.evaluate(combined)
        mask = result.values.astype(np.bool_) & result.validity
        cand_left = [i for i, keep in zip(cand_left, mask.tolist(), strict=True) if keep]
        cand_right = [j for j, keep in zip(cand_right, mask.tolist(), strict=True) if keep]

    final_left = list(cand_left)
    final_right = list(cand_right)

    if how == "LEFT":
        matched = set(cand_left)
        for i in range(left.num_rows):
            if i not in matched:
                final_left.append(i)
                final_right.append(-1)

    return _combine(
        left,
        right,
        np.array(final_left, dtype=np.int64),
        np.array(final_right, dtype=np.int64),
    )


class HashJoin(Operator):
    """Equi-join by building a hash table on the right input and probing it."""

    def __init__(
        self,
        left: Operator,
        right: Operator,
        left_keys: Sequence[Expression],
        right_keys: Sequence[Expression],
        how: str = "INNER",
        residual: Expression | None = None,
    ) -> None:
        if not left_keys or len(left_keys) != len(right_keys):
            raise ValueError("hash join needs matching, non-empty key lists")
        self.left = left
        self.right = right
        self.left_keys = list(left_keys)
        self.right_keys = list(right_keys)
        self.how = how
        self.residual = residual

    def execute(self) -> Table:
        left_table = self.left.execute()
        right_table = self.right.execute()

        right_keys = [_key_rows(k.evaluate(right_table)) for k in self.right_keys]
        index: dict[tuple[object, ...], list[int]] = {}
        for j in range(right_table.num_rows):
            key = tuple(col[j] for col in right_keys)
            if any(v is None for v in key):  # NULL keys never match
                continue
            index.setdefault(key, []).append(j)

        left_keys = [_key_rows(k.evaluate(left_table)) for k in self.left_keys]
        cand_left: list[int] = []
        cand_right: list[int] = []
        for i in range(left_table.num_rows):
            key = tuple(col[i] for col in left_keys)
            if any(v is None for v in key):
                continue
            for j in index.get(key, ()):
                cand_left.append(i)
                cand_right.append(j)

        return _finalize(left_table, right_table, cand_left, cand_right, self.how, self.residual)

    def schema(self) -> Schema:
        return {**self.left.schema(), **self.right.schema()}

    @property
    def children(self) -> Sequence[Operator]:
        return (self.left, self.right)

    def _describe(self) -> str:
        keys = ", ".join(
            f"{lk.output_name()}={rk.output_name()}"
            for lk, rk in zip(self.left_keys, self.right_keys, strict=True)
        )
        return f"HashJoin({self.how}, on=[{keys}])"


class NestedLoopJoin(Operator):
    """Join by evaluating ``predicate`` over every pair of rows."""

    def __init__(
        self,
        left: Operator,
        right: Operator,
        predicate: Expression | None,
        how: str = "INNER",
    ) -> None:
        self.left = left
        self.right = right
        self.predicate = predicate
        self.how = how

    def execute(self) -> Table:
        left_table = self.left.execute()
        right_table = self.right.execute()
        nl, nr = left_table.num_rows, right_table.num_rows

        if nr == 0:
            cand_left: list[int] = []
            cand_right: list[int] = []
        else:
            cand_left = np.repeat(np.arange(nl), nr).tolist()
            cand_right = np.tile(np.arange(nr), nl).tolist()

        return _finalize(left_table, right_table, cand_left, cand_right, self.how, self.predicate)

    def schema(self) -> Schema:
        return {**self.left.schema(), **self.right.schema()}

    @property
    def children(self) -> Sequence[Operator]:
        return (self.left, self.right)

    def _describe(self) -> str:
        pred = self.predicate.output_name() if self.predicate is not None else "true"
        return f"NestedLoopJoin({self.how}, on={pred})"


def _key_rows(column: Column) -> list[object | None]:
    """Materialise a key column to Python values for hashing."""
    return column.to_pylist()
