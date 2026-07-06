"""Scalar function registry: row-wise functions callable from SQL.

Each function is vectorised (it maps whole columns to a whole column) and
null-aware. The registry keeps the kernel and a result-type rule together so
the planner can type a call without evaluating it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from prism.column import Column
from prism.types import DataType, is_numeric, numpy_dtype, unify

Kernel = Callable[[Sequence[Column], str], Column]
TypeRule = Callable[[Sequence[DataType]], DataType]


@dataclass(frozen=True)
class ScalarFn:
    """A registered scalar function."""

    name: str
    min_args: int
    max_args: int | None  # None means variadic
    type_rule: TypeRule
    kernel: Kernel

    def check_arity(self, count: int) -> None:
        if count < self.min_args or (self.max_args is not None and count > self.max_args):
            want = self.min_args if self.max_args == self.min_args else f"{self.min_args}+"
            raise TypeError(f"{self.name} takes {want} argument(s), got {count}")


_REGISTRY: dict[str, ScalarFn] = {}


def register(fn: ScalarFn) -> None:
    _REGISTRY[fn.name] = fn


def is_scalar_function(name: str) -> bool:
    return name.upper() in _REGISTRY


def get(name: str) -> ScalarFn:
    try:
        return _REGISTRY[name.upper()]
    except KeyError:
        raise KeyError(f"unknown function {name!r}") from None


# ----------------------------------------------------------------------
# kernels
# ----------------------------------------------------------------------


def _upper(cols: Sequence[Column], name: str) -> Column:
    src = cols[0]
    values = np.array([s.upper() if isinstance(s, str) else s for s in src.values], dtype=object)
    return Column(name, DataType.TEXT, values, src.validity.copy())


def _lower(cols: Sequence[Column], name: str) -> Column:
    src = cols[0]
    values = np.array([s.lower() if isinstance(s, str) else s for s in src.values], dtype=object)
    return Column(name, DataType.TEXT, values, src.validity.copy())


def _length(cols: Sequence[Column], name: str) -> Column:
    src = cols[0]
    values = np.array([len(s) if isinstance(s, str) else 0 for s in src.values], dtype=np.int64)
    return Column(name, DataType.INTEGER, values, src.validity.copy())


def _abs(cols: Sequence[Column], name: str) -> Column:
    src = cols[0]
    return Column(name, src.dtype, np.abs(src.values), src.validity.copy())


def _round(cols: Sequence[Column], name: str) -> Column:
    src = cols[0]
    ndigits = 0
    if len(cols) == 2:
        ndigits = int(cols[1].values[0]) if len(cols[1].values) else 0
    values = np.round(src.values.astype(np.float64), ndigits)
    return Column(name, DataType.FLOAT, values, src.validity.copy())


def _coalesce(cols: Sequence[Column], name: str) -> Column:
    n = len(cols[0])
    result_type = cols[0].dtype
    for c in cols[1:]:
        result_type = unify(result_type, c.dtype)
    out_values: np.ndarray = np.empty(n, dtype=numpy_dtype(result_type))
    out_valid = np.zeros(n, dtype=np.bool_)
    for c in cols:
        take = c.validity & ~out_valid
        if take.any():
            out_values[take] = c.values[take]
            out_valid[take] = True
    return Column(name, result_type, out_values, out_valid)


def _text_rule(_: Sequence[DataType]) -> DataType:
    return DataType.TEXT


def _int_rule(_: Sequence[DataType]) -> DataType:
    return DataType.INTEGER


def _float_rule(_: Sequence[DataType]) -> DataType:
    return DataType.FLOAT


def _same_numeric_rule(types: Sequence[DataType]) -> DataType:
    if not is_numeric(types[0]):
        raise TypeError(f"ABS requires a numeric argument, got {types[0]}")
    return types[0]


def _coalesce_rule(types: Sequence[DataType]) -> DataType:
    result = types[0]
    for t in types[1:]:
        result = unify(result, t)
    return result


register(ScalarFn("UPPER", 1, 1, _text_rule, _upper))
register(ScalarFn("LOWER", 1, 1, _text_rule, _lower))
register(ScalarFn("LENGTH", 1, 1, _int_rule, _length))
register(ScalarFn("ABS", 1, 1, _same_numeric_rule, _abs))
register(ScalarFn("ROUND", 1, 2, _float_rule, _round))
register(ScalarFn("COALESCE", 1, None, _coalesce_rule, _coalesce))
