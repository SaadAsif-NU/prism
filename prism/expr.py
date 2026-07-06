"""Expressions: a typed, vectorised expression tree with a fluent builder.

An :class:`Expression` computes one output column from an input table. The same
tree is produced two ways: directly in Python via the builder API
(``col("age") > 30``) and by the SQL parser (Day 2). Because both targets share
this representation, the planner, optimizer, and executor only ever deal with
expressions, never with SQL syntax.

Evaluation is vectorised: every node returns a whole :class:`Column`, operating
on the input's values buffers in bulk rather than row by row. Nullability
follows SQL semantics, including three-valued logic for ``AND``/``OR``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from prism.column import Column
from prism.table import Table
from prism.types import DataType, is_numeric, numpy_dtype, unify

Schema = dict[str, DataType]


class BinOp(str, Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    EQ = "="
    NE = "<>"
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    AND = "AND"
    OR = "OR"


_ARITHMETIC = frozenset({BinOp.ADD, BinOp.SUB, BinOp.MUL, BinOp.DIV, BinOp.MOD})
_COMPARISON = frozenset({BinOp.EQ, BinOp.NE, BinOp.LT, BinOp.LE, BinOp.GT, BinOp.GE})
_LOGICAL = frozenset({BinOp.AND, BinOp.OR})


class Expression(ABC):
    """Base class for all expression tree nodes."""

    @abstractmethod
    def evaluate(self, table: Table) -> Column:
        """Compute this expression over ``table`` and return a column."""

    @abstractmethod
    def resolve_type(self, schema: Schema) -> DataType:
        """Return the result type given the input column types."""

    @abstractmethod
    def references(self) -> set[str]:
        """Column names this expression reads (drives projection pushdown)."""

    @abstractmethod
    def output_name(self) -> str:
        """Default column name when this expression is projected."""

    # -- fluent builder: arithmetic ------------------------------------

    def __add__(self, other: object) -> Expression:
        return BinaryExpr(BinOp.ADD, self, _lift(other))

    def __sub__(self, other: object) -> Expression:
        return BinaryExpr(BinOp.SUB, self, _lift(other))

    def __mul__(self, other: object) -> Expression:
        return BinaryExpr(BinOp.MUL, self, _lift(other))

    def __truediv__(self, other: object) -> Expression:
        return BinaryExpr(BinOp.DIV, self, _lift(other))

    def __mod__(self, other: object) -> Expression:
        return BinaryExpr(BinOp.MOD, self, _lift(other))

    def __neg__(self) -> Expression:
        return NegateExpr(self)

    # -- fluent builder: comparison ------------------------------------

    def __eq__(self, other: object) -> Expression:  # type: ignore[override]
        return BinaryExpr(BinOp.EQ, self, _lift(other))

    def __ne__(self, other: object) -> Expression:  # type: ignore[override]
        return BinaryExpr(BinOp.NE, self, _lift(other))

    def __lt__(self, other: object) -> Expression:
        return BinaryExpr(BinOp.LT, self, _lift(other))

    def __le__(self, other: object) -> Expression:
        return BinaryExpr(BinOp.LE, self, _lift(other))

    def __gt__(self, other: object) -> Expression:
        return BinaryExpr(BinOp.GT, self, _lift(other))

    def __ge__(self, other: object) -> Expression:
        return BinaryExpr(BinOp.GE, self, _lift(other))

    # -- fluent builder: logical ---------------------------------------

    def __and__(self, other: object) -> Expression:
        return BinaryExpr(BinOp.AND, self, _lift(other))

    def __or__(self, other: object) -> Expression:
        return BinaryExpr(BinOp.OR, self, _lift(other))

    def __invert__(self) -> Expression:
        return NotExpr(self)

    def is_null(self) -> Expression:
        return IsNullExpr(self, negated=False)

    def is_not_null(self) -> Expression:
        return IsNullExpr(self, negated=True)

    def alias(self, name: str) -> AliasExpr:
        return AliasExpr(self, name)

    # Expressions are identity-hashable despite the __eq__ override so they can
    # live in sets (e.g. during optimization).
    __hash__ = object.__hash__


class ColumnRef(Expression):
    """A reference to a column by name."""

    def __init__(self, name: str) -> None:
        self.name = name

    def evaluate(self, table: Table) -> Column:
        return table.column(self.name)

    def resolve_type(self, schema: Schema) -> DataType:
        if self.name not in schema:
            raise KeyError(f"unknown column {self.name!r}")
        return schema[self.name]

    def references(self) -> set[str]:
        return {self.name}

    def output_name(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


class Literal(Expression):
    """A constant scalar (``None`` for the SQL NULL literal)."""

    def __init__(self, value: object | None, dtype: DataType | None = None) -> None:
        self.value = value
        self.dtype = dtype if dtype is not None else _infer_literal_type(value)

    def evaluate(self, table: Table) -> Column:
        n = table.num_rows
        data: list[object | None] = [self.value] * n
        return Column.from_pylist(self.output_name(), data, self.dtype)

    def resolve_type(self, schema: Schema) -> DataType:
        return self.dtype

    def references(self) -> set[str]:
        return set()

    def output_name(self) -> str:
        return "NULL" if self.value is None else repr(self.value)

    def __repr__(self) -> str:
        return "NULL" if self.value is None else repr(self.value)


class AliasExpr(Expression):
    """Wraps an expression to give its projected column a chosen name."""

    def __init__(self, inner: Expression, name: str) -> None:
        self.inner = inner
        self.name = name

    def evaluate(self, table: Table) -> Column:
        return self.inner.evaluate(table).rename(self.name)

    def resolve_type(self, schema: Schema) -> DataType:
        return self.inner.resolve_type(schema)

    def references(self) -> set[str]:
        return self.inner.references()

    def output_name(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.inner!r} AS {self.name}"


class BinaryExpr(Expression):
    """A binary operation: arithmetic, comparison, or logical."""

    def __init__(self, op: BinOp, left: Expression, right: Expression) -> None:
        self.op = op
        self.left = left
        self.right = right

    def evaluate(self, table: Table) -> Column:
        left = self.left.evaluate(table)
        right = self.right.evaluate(table)
        if self.op in _LOGICAL:
            return _eval_logical(self.op, left, right, self.output_name())
        if self.op in _COMPARISON:
            return _eval_comparison(self.op, left, right, self.output_name())
        return _eval_arithmetic(self.op, left, right, self.output_name())

    def resolve_type(self, schema: Schema) -> DataType:
        lt = self.left.resolve_type(schema)
        rt = self.right.resolve_type(schema)
        if self.op in _LOGICAL or self.op in _COMPARISON:
            return DataType.BOOLEAN
        if self.op is BinOp.DIV:
            return DataType.FLOAT
        if is_numeric(lt) and is_numeric(rt):
            return DataType.FLOAT if DataType.FLOAT in (lt, rt) else DataType.INTEGER
        raise TypeError(f"arithmetic {self.op.value} not defined for {lt}, {rt}")

    def references(self) -> set[str]:
        return self.left.references() | self.right.references()

    def output_name(self) -> str:
        return f"({self.left.output_name()} {self.op.value} {self.right.output_name()})"

    def __repr__(self) -> str:
        return f"({self.left!r} {self.op.value} {self.right!r})"


class NotExpr(Expression):
    """Logical negation with three-valued semantics (NOT NULL is NULL)."""

    def __init__(self, operand: Expression) -> None:
        self.operand = operand

    def evaluate(self, table: Table) -> Column:
        col = self.operand.evaluate(table)
        values = ~col.values.astype(np.bool_)
        return Column(self.output_name(), DataType.BOOLEAN, values, col.validity.copy())

    def resolve_type(self, schema: Schema) -> DataType:
        return DataType.BOOLEAN

    def references(self) -> set[str]:
        return self.operand.references()

    def output_name(self) -> str:
        return f"(NOT {self.operand.output_name()})"

    def __repr__(self) -> str:
        return f"(NOT {self.operand!r})"


class NegateExpr(Expression):
    """Arithmetic negation of a numeric expression."""

    def __init__(self, operand: Expression) -> None:
        self.operand = operand

    def evaluate(self, table: Table) -> Column:
        col = self.operand.evaluate(table)
        return Column(self.output_name(), col.dtype, -col.values, col.validity.copy())

    def resolve_type(self, schema: Schema) -> DataType:
        dtype = self.operand.resolve_type(schema)
        if not is_numeric(dtype):
            raise TypeError(f"cannot negate non-numeric {dtype}")
        return dtype

    def references(self) -> set[str]:
        return self.operand.references()

    def output_name(self) -> str:
        return f"(-{self.operand.output_name()})"

    def __repr__(self) -> str:
        return f"(-{self.operand!r})"


class IsNullExpr(Expression):
    """``IS NULL`` / ``IS NOT NULL``; the result itself is never null."""

    def __init__(self, operand: Expression, negated: bool) -> None:
        self.operand = operand
        self.negated = negated

    def evaluate(self, table: Table) -> Column:
        col = self.operand.evaluate(table)
        present = col.validity
        values = present.copy() if self.negated else ~present
        validity = np.ones(len(values), dtype=np.bool_)
        return Column(self.output_name(), DataType.BOOLEAN, values, validity)

    def resolve_type(self, schema: Schema) -> DataType:
        return DataType.BOOLEAN

    def references(self) -> set[str]:
        return self.operand.references()

    def output_name(self) -> str:
        kind = "IS NOT NULL" if self.negated else "IS NULL"
        return f"({self.operand.output_name()} {kind})"

    def __repr__(self) -> str:
        return self.output_name()


class ScalarFunction(Expression):
    """A call to a registered row-wise function (e.g. ``UPPER(name)``)."""

    def __init__(self, name: str, args: list[Expression]) -> None:
        self.name = name.upper()
        self.args = args

    def evaluate(self, table: Table) -> Column:
        from prism.functions import get

        fn = get(self.name)
        fn.check_arity(len(self.args))
        columns = [arg.evaluate(table) for arg in self.args]
        return fn.kernel(columns, self.output_name())

    def resolve_type(self, schema: Schema) -> DataType:
        from prism.functions import get

        fn = get(self.name)
        fn.check_arity(len(self.args))
        return fn.type_rule([arg.resolve_type(schema) for arg in self.args])

    def references(self) -> set[str]:
        refs: set[str] = set()
        for arg in self.args:
            refs |= arg.references()
        return refs

    def output_name(self) -> str:
        inner = ", ".join(arg.output_name() for arg in self.args)
        return f"{self.name}({inner})"

    def __repr__(self) -> str:
        return self.output_name()


# ----------------------------------------------------------------------
# builder helpers
# ----------------------------------------------------------------------


def col(name: str) -> ColumnRef:
    """Reference a column by name."""
    return ColumnRef(name)


def lit(value: object | None) -> Literal:
    """A constant scalar literal."""
    return Literal(value)


def _lift(value: object) -> Expression:
    """Promote a raw Python value in the builder API into a Literal."""
    if isinstance(value, Expression):
        return value
    return Literal(value)


def _infer_literal_type(value: object | None) -> DataType:
    if value is None:
        return DataType.NULL
    if isinstance(value, bool):
        return DataType.BOOLEAN
    if isinstance(value, int):
        return DataType.INTEGER
    if isinstance(value, float):
        return DataType.FLOAT
    return DataType.TEXT


# ----------------------------------------------------------------------
# vectorised evaluation kernels
# ----------------------------------------------------------------------


def _eval_arithmetic(op: BinOp, left: Column, right: Column, name: str) -> Column:
    validity = left.validity & right.validity
    lv, rv = left.values, right.values
    if op is BinOp.ADD:
        values = lv + rv
    elif op is BinOp.SUB:
        values = lv - rv
    elif op is BinOp.MUL:
        values = lv * rv
    elif op is BinOp.MOD:
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(rv != 0, lv % np.where(rv == 0, 1, rv), 0)
        validity = validity & (rv != 0)
    else:  # DIV -> always float, nulls where divisor is zero
        lf = lv.astype(np.float64)
        rf = rv.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(rf != 0, lf / np.where(rf == 0, 1, rf), 0.0)
        validity = validity & (rf != 0)
    result_type = DataType.FLOAT if op is BinOp.DIV else _numeric_result(left.dtype, right.dtype)
    return Column(name, result_type, values.astype(numpy_dtype(result_type)), validity)


def _eval_comparison(op: BinOp, left: Column, right: Column, name: str) -> Column:
    validity = left.validity & right.validity
    lv, rv = left.values, right.values
    if op is BinOp.EQ:
        values = lv == rv
    elif op is BinOp.NE:
        values = lv != rv
    elif op is BinOp.LT:
        values = lv < rv
    elif op is BinOp.LE:
        values = lv <= rv
    elif op is BinOp.GT:
        values = lv > rv
    else:  # GE
        values = lv >= rv
    return Column(name, DataType.BOOLEAN, np.asarray(values, dtype=np.bool_), validity)


def _eval_logical(op: BinOp, left: Column, right: Column, name: str) -> Column:
    lv = left.values.astype(np.bool_)
    rv = right.values.astype(np.bool_)
    lvalid, rvalid = left.validity, right.validity
    both_valid = lvalid & rvalid
    if op is BinOp.AND:
        values = lv & rv
        # Known-false short-circuits an unknown operand to FALSE.
        validity = both_valid | (lvalid & ~lv) | (rvalid & ~rv)
    else:  # OR: known-true short-circuits an unknown operand to TRUE.
        values = lv | rv
        validity = both_valid | (lvalid & lv) | (rvalid & rv)
    return Column(name, DataType.BOOLEAN, values, validity)


def _numeric_result(left: DataType, right: DataType) -> DataType:
    if not (is_numeric(left) and is_numeric(right)):
        raise TypeError(f"arithmetic not defined for {left}, {right}")
    return DataType.FLOAT if DataType.FLOAT in (left, right) else DataType.INTEGER


def schema_of(table: Table) -> Schema:
    """Build a name to type schema mapping from a table."""
    return dict(table.schema)


def unify_types(left: DataType, right: DataType) -> DataType:
    """Re-exported for planner use."""
    return unify(left, right)
