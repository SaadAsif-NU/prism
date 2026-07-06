"""Aggregate functions and their grouped reductions.

An :class:`AggregateSpec` describes one aggregate to compute (the function, its
argument expression, whether DISTINCT applies, and the output column name).
:func:`reduce_grouped` evaluates it against a table given a per-row group id,
producing one value per group. The common path is vectorised with NumPy's
scatter-reductions; DISTINCT and TEXT MIN/MAX fall back to per-group Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from prism.column import Column
from prism.expr import Expression
from prism.table import Table
from prism.types import DataType, numpy_dtype


class AggFunc(str, Enum):
    COUNT = "COUNT"
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"


_AGG_NAMES = frozenset(f.value for f in AggFunc)


def is_aggregate(name: str) -> bool:
    return name.upper() in _AGG_NAMES


class AggregateExpr(Expression):
    """An aggregate call in an expression tree (e.g. ``SUM(salary)``).

    It carries the scalar argument expression it reduces over. It is never
    evaluated per row: the :class:`~prism.exec.aggregate.HashAggregate`
    operator recognises it and computes it across a whole group instead.
    """

    def __init__(self, func: AggFunc, arg: Expression | None, distinct: bool = False) -> None:
        self.func = func
        self.arg = arg
        self.distinct = distinct

    def evaluate(self, table: Table) -> Column:
        raise TypeError(
            f"{self.output_name()} is an aggregate and cannot be evaluated per row; "
            "it must be computed by a grouping operator"
        )

    def resolve_type(self, schema: dict[str, DataType]) -> DataType:
        arg_type = self.arg.resolve_type(schema) if self.arg is not None else None
        return agg_result_type(self.func, arg_type)

    def references(self) -> set[str]:
        return self.arg.references() if self.arg is not None else set()

    def output_name(self) -> str:
        if self.arg is None:
            return f"{self.func.value}(*)"
        inner = self.arg.output_name()
        if self.distinct:
            inner = f"DISTINCT {inner}"
        return f"{self.func.value}({inner})"

    def __repr__(self) -> str:
        return self.output_name()


def agg_result_type(func: AggFunc, arg_type: DataType | None) -> DataType:
    """The output type of an aggregate given its argument type."""
    if func is AggFunc.COUNT:
        return DataType.INTEGER
    if func is AggFunc.AVG:
        return DataType.FLOAT
    if arg_type is None:
        raise TypeError(f"{func.value} requires an argument")
    if func is AggFunc.SUM:
        return DataType.FLOAT if arg_type is DataType.FLOAT else DataType.INTEGER
    return arg_type  # MIN / MAX keep the argument type


@dataclass(frozen=True)
class AggregateSpec:
    """One aggregate to compute in a grouped or global aggregation."""

    func: AggFunc
    arg: Expression | None  # None only for COUNT(*)
    distinct: bool
    output_name: str
    output_type: DataType


def reduce_grouped(
    spec: AggregateSpec,
    table: Table,
    group_ids: np.ndarray,
    n_groups: int,
) -> Column:
    """Compute one aggregate column, one value per group."""
    arg = spec.arg.evaluate(table) if spec.arg is not None else None

    if spec.distinct:
        return _reduce_distinct(spec, arg, group_ids, n_groups)
    if spec.func is AggFunc.COUNT:
        return _reduce_count(spec, arg, group_ids, n_groups)
    assert arg is not None
    if spec.func in (AggFunc.SUM, AggFunc.AVG):
        return _reduce_sum_avg(spec, arg, group_ids, n_groups)
    return _reduce_minmax(spec, arg, group_ids, n_groups)


def _valid_counts(group_ids: np.ndarray, validity: np.ndarray, n_groups: int) -> np.ndarray:
    return np.bincount(group_ids, weights=validity.astype(np.float64), minlength=n_groups)


def _reduce_count(
    spec: AggregateSpec, arg: Column | None, group_ids: np.ndarray, n_groups: int
) -> Column:
    if arg is None:  # COUNT(*): every row counts
        counts = np.bincount(group_ids, minlength=n_groups).astype(np.int64)
    else:  # COUNT(expr): only non-null rows count
        counts = _valid_counts(group_ids, arg.validity, n_groups).astype(np.int64)
    return Column.from_values(spec.output_name, DataType.INTEGER, counts)


def _reduce_sum_avg(
    spec: AggregateSpec, arg: Column, group_ids: np.ndarray, n_groups: int
) -> Column:
    values = arg.values.astype(np.float64)
    contrib = np.where(arg.validity, values, 0.0)
    sums = np.bincount(group_ids, weights=contrib, minlength=n_groups)
    counts = _valid_counts(group_ids, arg.validity, n_groups)
    group_valid = counts > 0

    if spec.func is AggFunc.AVG:
        with np.errstate(invalid="ignore"):
            avg = np.where(group_valid, sums / np.where(counts == 0, 1, counts), 0.0)
        return Column(spec.output_name, DataType.FLOAT, avg, group_valid)

    out: np.ndarray = sums.astype(numpy_dtype(spec.output_type))
    return Column(spec.output_name, spec.output_type, out, group_valid)


def _reduce_minmax(
    spec: AggregateSpec, arg: Column, group_ids: np.ndarray, n_groups: int
) -> Column:
    counts = _valid_counts(group_ids, arg.validity, n_groups)
    group_valid = counts > 0

    if arg.dtype is DataType.TEXT:
        return _reduce_minmax_text(spec, arg, group_ids, n_groups, group_valid)

    values = arg.values.astype(np.float64)
    fill = np.inf if spec.func is AggFunc.MIN else -np.inf
    acc = np.full(n_groups, fill, dtype=np.float64)
    gids = group_ids[arg.validity]
    vals = values[arg.validity]
    if spec.func is AggFunc.MIN:
        np.minimum.at(acc, gids, vals)
    else:
        np.maximum.at(acc, gids, vals)
    acc = np.where(group_valid, acc, 0.0)
    return Column(
        spec.output_name, spec.output_type, acc.astype(numpy_dtype(spec.output_type)), group_valid
    )


def _reduce_minmax_text(
    spec: AggregateSpec,
    arg: Column,
    group_ids: np.ndarray,
    n_groups: int,
    group_valid: np.ndarray,
) -> Column:
    best: list[object | None] = [None] * n_groups
    take_min = spec.func is AggFunc.MIN
    for gid, value, valid in zip(
        group_ids.tolist(), arg.values, arg.validity.tolist(), strict=True
    ):
        if not valid:
            continue
        current = best[gid]
        if current is None or (value < current if take_min else value > current):
            best[gid] = value
    return Column.from_pylist(spec.output_name, best, DataType.TEXT)


def _reduce_distinct(
    spec: AggregateSpec, arg: Column | None, group_ids: np.ndarray, n_groups: int
) -> Column:
    """Per-group reduction over the distinct non-null argument values."""
    if arg is None:
        raise TypeError("DISTINCT requires an argument")
    buckets: list[set[object]] = [set() for _ in range(n_groups)]
    for gid, value, valid in zip(
        group_ids.tolist(), arg.values.tolist(), arg.validity.tolist(), strict=True
    ):
        if valid:
            buckets[gid].add(value)

    results: list[object | None] = []
    for values in buckets:
        results.append(_apply_scalar(spec.func, values))
    return Column.from_pylist(spec.output_name, results, spec.output_type)


def _apply_scalar(func: AggFunc, values: set[object]) -> object | None:
    if func is AggFunc.COUNT:
        return len(values)
    if not values:
        return None
    if func is AggFunc.SUM:
        return sum(values)  # type: ignore[arg-type]
    if func is AggFunc.AVG:
        return sum(values) / len(values)  # type: ignore[arg-type]
    if func is AggFunc.MIN:
        return min(values)  # type: ignore[type-var]
    return max(values)  # type: ignore[type-var]
