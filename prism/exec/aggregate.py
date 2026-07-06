"""Grouping operators: hash aggregation and DISTINCT.

:class:`HashAggregate` partitions rows into groups by hashing the group-key
values, then reduces each aggregate over the rows of every group. With no group
keys it still emits exactly one row, matching SQL's global aggregate.
:class:`Distinct` deduplicates whole rows while preserving first-seen order.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from prism.aggregate import AggregateSpec, reduce_grouped
from prism.column import Column
from prism.exec.operators import Operator
from prism.expr import Expression, Schema
from prism.table import Table


def factorize(group_columns: Sequence[Column]) -> tuple[np.ndarray, int, np.ndarray]:
    """Assign each row a group id from its group-key values.

    Returns the per-row group ids, the number of distinct groups, and the row
    index where each group first appeared (used to materialise the key columns).
    Groups are numbered in first-appearance order, so results are deterministic.
    NULL keys collapse into one group, as SQL requires.
    """
    n = len(group_columns[0]) if group_columns else 0
    key_lists = [c.to_pylist() for c in group_columns]
    mapping: dict[tuple[object, ...], int] = {}
    group_ids = np.empty(n, dtype=np.int64)
    first_indices: list[int] = []
    for i in range(n):
        key = tuple(col[i] for col in key_lists)
        gid = mapping.get(key)
        if gid is None:
            gid = len(mapping)
            mapping[key] = gid
            first_indices.append(i)
        group_ids[i] = gid
    return group_ids, len(mapping), np.array(first_indices, dtype=np.int64)


class HashAggregate(Operator):
    """Groups rows by ``group_exprs`` and computes ``specs`` over each group."""

    def __init__(
        self,
        child: Operator,
        group_exprs: Sequence[Expression],
        specs: Sequence[AggregateSpec],
    ) -> None:
        self.child = child
        self.group_exprs = list(group_exprs)
        self.specs = list(specs)

    def execute(self) -> Table:
        table = self.child.execute()

        if self.group_exprs:
            group_columns = [e.evaluate(table) for e in self.group_exprs]
            group_ids, n_groups, first_indices = factorize(group_columns)
            key_columns = [c.take(first_indices) for c in group_columns]
        else:
            # Global aggregate: one group covering every row (and one output
            # row even when the input is empty).
            group_ids = np.zeros(table.num_rows, dtype=np.int64)
            n_groups = 1
            key_columns = []

        agg_columns = [reduce_grouped(spec, table, group_ids, n_groups) for spec in self.specs]
        return Table(key_columns + agg_columns)

    def schema(self) -> Schema:
        child_schema = self.child.schema()
        out: Schema = {
            expr.output_name(): expr.resolve_type(child_schema) for expr in self.group_exprs
        }
        for spec in self.specs:
            out[spec.output_name] = spec.output_type
        return out

    @property
    def children(self) -> Sequence[Operator]:
        return (self.child,)

    def _describe(self) -> str:
        keys = ", ".join(e.output_name() for e in self.group_exprs) or "(global)"
        aggs = ", ".join(s.output_name for s in self.specs)
        return f"HashAggregate(keys=[{keys}], aggs=[{aggs}])"


class Distinct(Operator):
    """Removes duplicate rows, keeping the first occurrence of each."""

    def __init__(self, child: Operator) -> None:
        self.child = child

    def execute(self) -> Table:
        table = self.child.execute()
        key_lists = [c.to_pylist() for c in table.columns]
        seen: set[tuple[object, ...]] = set()
        keep: list[int] = []
        for i in range(table.num_rows):
            key = tuple(col[i] for col in key_lists)
            if key not in seen:
                seen.add(key)
                keep.append(i)
        return table.take(np.array(keep, dtype=np.int64))

    def schema(self) -> Schema:
        return self.child.schema()

    @property
    def children(self) -> Sequence[Operator]:
        return (self.child,)

    def _describe(self) -> str:
        return "Distinct"
