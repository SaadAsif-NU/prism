"""The abstract syntax tree the parser produces.

This is a pure *syntactic* representation: it mirrors what was written, not how
it will run. Column references keep their optional table qualifier, aggregate
and scalar calls are undifferentiated ``FunctionCall`` nodes, and nothing is
type-checked yet. The planner (:mod:`prism.plan`) binds this tree to executable
:mod:`prism.expr` expressions and physical operators.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from prism.types import DataType

# ----------------------------------------------------------------------
# expressions
# ----------------------------------------------------------------------


class Expr:
    """Base class for SQL expression AST nodes."""


@dataclass(frozen=True)
class ColumnRef(Expr):
    """A column reference, optionally qualified by a table name or alias."""

    name: str
    table: str | None = None


@dataclass(frozen=True)
class Literal(Expr):
    """A constant scalar literal."""

    value: object | None
    dtype: DataType


@dataclass(frozen=True)
class BinaryOp(Expr):
    """An infix binary operation (arithmetic, comparison, or logical)."""

    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class UnaryOp(Expr):
    """A prefix unary operation: ``NOT expr`` or ``-expr``."""

    op: str
    operand: Expr


@dataclass(frozen=True)
class IsNull(Expr):
    """An ``IS NULL`` or ``IS NOT NULL`` test."""

    operand: Expr
    negated: bool


@dataclass(frozen=True)
class FunctionCall(Expr):
    """A function call: an aggregate (``SUM(x)``) or a scalar (``UPPER(s)``).

    ``star`` marks the special ``COUNT(*)`` form; ``distinct`` marks
    ``COUNT(DISTINCT x)`` and friends.
    """

    name: str
    args: tuple[Expr, ...] = ()
    distinct: bool = False
    star: bool = False


@dataclass(frozen=True)
class Star(Expr):
    """The ``*`` (or ``t.*``) projection wildcard."""

    table: str | None = None


# ----------------------------------------------------------------------
# select statement
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SelectItem:
    """One entry in the SELECT list: an expression with an optional alias."""

    expr: Expr
    alias: str | None = None


@dataclass(frozen=True)
class TableRef:
    """A table named in FROM, with an optional alias."""

    name: str
    alias: str | None = None

    @property
    def bound_name(self) -> str:
        """The name this reference is addressed by (alias if present)."""
        return self.alias if self.alias is not None else self.name


@dataclass(frozen=True)
class Join:
    """A join of the current relation with ``right`` on a predicate."""

    kind: str  # "INNER" or "LEFT"
    right: TableRef
    on: Expr


@dataclass(frozen=True)
class OrderKey:
    """One ORDER BY term."""

    expr: Expr
    ascending: bool = True
    nulls_first: bool | None = None


@dataclass(frozen=True)
class SelectStatement:
    """A parsed ``SELECT`` query."""

    items: tuple[SelectItem, ...]
    from_table: TableRef | None = None
    joins: tuple[Join, ...] = ()
    where: Expr | None = None
    group_by: tuple[Expr, ...] = ()
    having: Expr | None = None
    order_by: tuple[OrderKey, ...] = ()
    limit: int | None = None
    offset: int = 0
    distinct: bool = False
    # Extra tables listed in FROM (comma-separated) become cross joins.
    extra_tables: tuple[TableRef, ...] = field(default_factory=tuple)
