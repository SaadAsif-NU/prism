"""The planner: bind a parsed ``SELECT`` to an executable operator tree.

This is where syntax becomes execution. The planner resolves column references
against the tables in ``FROM`` (qualifying names across joins), lowers SQL
expression nodes to :mod:`prism.expr` expressions, extracts equi-join keys from
``ON`` clauses, and turns ``GROUP BY`` / aggregates into a
:class:`~prism.exec.aggregate.HashAggregate`. The result is the same kind of
operator tree the fluent :class:`~prism.relation.Relation` API builds.
"""

from __future__ import annotations

from collections import Counter

from prism.aggregate import AggFunc, AggregateSpec, agg_result_type, is_aggregate
from prism.column import Column
from prism.exec.aggregate import Distinct, HashAggregate
from prism.exec.join import HashJoin, NestedLoopJoin
from prism.exec.operators import Filter, Limit, Operator, Project, Scan, Sort, SortKey
from prism.expr import (
    BinaryExpr,
    BinOp,
    Expression,
    IsNullExpr,
    Literal,
    NegateExpr,
    NotExpr,
    ScalarFunction,
    col,
)
from prism.functions import is_scalar_function
from prism.sql import ast, parse
from prism.storage.catalog import Catalog
from prism.table import Table
from prism.types import DataType


class PlanError(ValueError):
    """Raised when a statement is syntactically valid but cannot be planned."""


def plan(statement: ast.SelectStatement | str, catalog: Catalog) -> Operator:
    """Plan a statement (already parsed, or SQL text) against ``catalog``."""
    stmt = parse(statement) if isinstance(statement, str) else statement
    return _Planner(catalog).plan(stmt)


# ----------------------------------------------------------------------
# name resolution scope
# ----------------------------------------------------------------------


class _Scope:
    """Resolves column references to the operator tree's output column names."""

    def __init__(self) -> None:
        self.columns: list[str] = []
        self.by_qualified: dict[tuple[str, str], str] = {}
        self.by_bare: dict[str, str | None] = {}
        self.relation_columns: dict[str, list[str]] = {}

    def resolve(self, ref: ast.ColumnRef) -> str:
        if ref.table is not None:
            key = (ref.table, ref.name)
            if key not in self.by_qualified:
                raise PlanError(f"unknown column {ref.table}.{ref.name}")
            return self.by_qualified[key]
        if ref.name not in self.by_bare:
            raise PlanError(f"unknown column {ref.name!r}")
        resolved = self.by_bare[ref.name]
        if resolved is None:
            raise PlanError(f"column {ref.name!r} is ambiguous; qualify it with a table")
        return resolved

    def star_columns(self, table: str | None) -> list[str]:
        if table is None:
            return list(self.columns)
        if table not in self.relation_columns:
            raise PlanError(f"unknown table {table!r} in {table}.*")
        return self.relation_columns[table]


# ----------------------------------------------------------------------
# aggregate collection
# ----------------------------------------------------------------------


class _AggregateContext:
    """Collects the distinct aggregates a query needs, assigning output names."""

    def __init__(self, planner: _Planner, scope: _Scope, base_schema: dict[str, DataType]) -> None:
        self.planner = planner
        self.scope = scope
        self.base_schema = base_schema
        self.specs: list[AggregateSpec] = []
        self._by_node: dict[ast.FunctionCall, str] = {}

    def register(self, node: ast.FunctionCall) -> str:
        if node in self._by_node:
            return self._by_node[node]
        func = AggFunc(node.name)
        if node.star:
            if func is not AggFunc.COUNT:
                raise PlanError("* is only valid as COUNT(*)")
            arg: Expression | None = None
            arg_type: DataType | None = None
        else:
            if len(node.args) != 1:
                raise PlanError(f"{func.value} takes exactly one argument")
            arg = self.planner.bind_scalar(node.args[0], self.scope)
            arg_type = arg.resolve_type(self.base_schema)
        output_name = f"__agg{len(self.specs)}"
        spec = AggregateSpec(func, arg, node.distinct, output_name, agg_result_type(func, arg_type))
        self.specs.append(spec)
        self._by_node[node] = output_name
        return output_name


# ----------------------------------------------------------------------
# planner
# ----------------------------------------------------------------------


class _Planner:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    def plan(self, stmt: ast.SelectStatement) -> Operator:
        if stmt.from_table is None:
            op: Operator = Scan(_unit_table(), "(constants)")
            scope = _Scope()
        else:
            op, scope = self._build_from(stmt)

        if stmt.where is not None:
            op = Filter(op, self.bind_scalar(stmt.where, scope))

        if stmt.group_by or _has_aggregate(stmt):
            return self._plan_aggregated(op, stmt, scope)
        return self._plan_simple(op, stmt, scope)

    # -- FROM / joins ---------------------------------------------------

    def _build_from(self, stmt: ast.SelectStatement) -> tuple[Operator, _Scope]:
        assert stmt.from_table is not None
        relations = [stmt.from_table, *(j.right for j in stmt.joins), *stmt.extra_tables]
        tables: dict[str, Table] = {}
        for ref in relations:
            if ref.bound_name in tables:
                raise PlanError(f"duplicate table name/alias {ref.bound_name!r}")
            if ref.name not in self.catalog:
                raise PlanError(f"unknown table {ref.name!r}")
            tables[ref.bound_name] = self.catalog.get(ref.name)

        counts: Counter[str] = Counter()
        for ref in relations:
            counts.update(tables[ref.bound_name].column_names)

        scope = _Scope()

        def global_name(bound: str, column: str) -> str:
            return column if counts[column] == 1 else f"{bound}.{column}"

        for ref in relations:
            bound = ref.bound_name
            names = []
            for column in tables[bound].column_names:
                g = global_name(bound, column)
                names.append(g)
                scope.columns.append(g)
                scope.by_qualified[(bound, column)] = g
                scope.by_bare[column] = None if column in scope.by_bare else g
            scope.relation_columns[bound] = names

        op = _scan_relation(stmt.from_table, tables[stmt.from_table.bound_name], global_name)
        left_names = set(scope.relation_columns[stmt.from_table.bound_name])

        for join in stmt.joins:
            right_op = _scan_relation(join.right, tables[join.right.bound_name], global_name)
            right_names = set(scope.relation_columns[join.right.bound_name])
            bound_on = self.bind_scalar(join.on, scope)
            left_keys, right_keys, residual = _split_equijoin(bound_on, left_names, right_names)
            if left_keys:
                op = HashJoin(op, right_op, left_keys, right_keys, join.kind, residual)
            else:
                op = NestedLoopJoin(op, right_op, bound_on, join.kind)
            left_names |= right_names

        for ref in stmt.extra_tables:
            right_op = _scan_relation(ref, tables[ref.bound_name], global_name)
            op = NestedLoopJoin(op, right_op, None, "INNER")
            left_names |= set(scope.relation_columns[ref.bound_name])

        return op, scope

    # -- simple (non-aggregated) queries --------------------------------

    def _plan_simple(self, child: Operator, stmt: ast.SelectStatement, scope: _Scope) -> Operator:
        alias_map = {i.alias: i.expr for i in stmt.items if i.alias is not None}
        op = child

        if stmt.order_by:
            keys = [
                SortKey(
                    self.bind_scalar(_alias_expr(ok.expr, alias_map), scope),
                    ok.ascending,
                    ok.nulls_first,
                )
                for ok in stmt.order_by
            ]
            op = Sort(op, keys)

        op = Project(op, self._projections(stmt.items, scope))
        if stmt.distinct:
            op = Distinct(op)
        return _limit(op, stmt)

    def _projections(self, items: tuple[ast.SelectItem, ...], scope: _Scope) -> list[Expression]:
        projections: list[Expression] = []
        for item in items:
            if isinstance(item.expr, ast.Star):
                projections.extend(col(name) for name in scope.star_columns(item.expr.table))
                continue
            bound = self.bind_scalar(item.expr, scope)
            projections.append(bound.alias(item.alias or _output_name(item.expr)))
        return projections

    # -- aggregated queries ---------------------------------------------

    def _plan_aggregated(
        self, child: Operator, stmt: ast.SelectStatement, scope: _Scope
    ) -> Operator:
        base_schema = child.schema()
        group_bound: list[Expression] = []
        group_key_map: dict[ast.Expr, str] = {}
        for i, group_expr in enumerate(stmt.group_by):
            name = f"__gk{i}"
            group_bound.append(self.bind_scalar(group_expr, scope).alias(name))
            group_key_map[group_expr] = name

        ctx = _AggregateContext(self, scope, base_schema)
        bound_items = [
            (
                self._bind_aggregated(item.expr, scope, group_key_map, ctx),
                item.alias or _output_name(item.expr),
            )
            for item in stmt.items
        ]
        bound_having = (
            self._bind_aggregated(stmt.having, scope, group_key_map, ctx)
            if stmt.having is not None
            else None
        )

        alias_map = {i.alias: i.expr for i in stmt.items if i.alias is not None}
        order_terms = [
            (
                self._bind_aggregated(_alias_expr(ok.expr, alias_map), scope, group_key_map, ctx),
                ok.ascending,
                ok.nulls_first,
            )
            for ok in stmt.order_by
        ]

        op: Operator = HashAggregate(child, group_bound, ctx.specs)
        if bound_having is not None:
            op = Filter(op, bound_having)
        if order_terms:
            op = Sort(op, [SortKey(e, asc, nf) for e, asc, nf in order_terms])

        op = Project(op, [e.alias(a) for e, a in bound_items])
        if stmt.distinct:
            op = Distinct(op)
        return _limit(op, stmt)

    def _bind_aggregated(
        self,
        node: ast.Expr,
        scope: _Scope,
        group_key_map: dict[ast.Expr, str],
        ctx: _AggregateContext,
    ) -> Expression:
        if node in group_key_map:
            return col(group_key_map[node])
        if isinstance(node, ast.FunctionCall) and is_aggregate(node.name):
            return col(ctx.register(node))
        if isinstance(node, ast.Literal):
            return Literal(node.value, node.dtype)
        if isinstance(node, ast.ColumnRef):
            raise PlanError(
                f"column {node.name!r} must appear in GROUP BY or be used in an aggregate"
            )
        if isinstance(node, ast.Star):
            raise PlanError("* is not allowed with GROUP BY or aggregates")
        return self._rebuild(node, lambda n: self._bind_aggregated(n, scope, group_key_map, ctx))

    # -- expression binding ---------------------------------------------

    def bind_scalar(self, node: ast.Expr, scope: _Scope) -> Expression:
        if isinstance(node, ast.ColumnRef):
            return col(scope.resolve(node))
        if isinstance(node, ast.Literal):
            return Literal(node.value, node.dtype)
        if isinstance(node, ast.FunctionCall):
            if is_aggregate(node.name):
                raise PlanError(f"aggregate {node.name} is not allowed here")
            if not is_scalar_function(node.name):
                raise PlanError(f"unknown function {node.name}")
            return ScalarFunction(node.name, [self.bind_scalar(a, scope) for a in node.args])
        if isinstance(node, ast.Star):
            raise PlanError("* is not allowed in this context")
        return self._rebuild(node, lambda n: self.bind_scalar(n, scope))

    def _rebuild(self, node: ast.Expr, bind: object) -> Expression:
        """Rebuild a compound node, binding children with ``bind``."""
        assert callable(bind)
        if isinstance(node, ast.BinaryOp):
            return BinaryExpr(BinOp(node.op), bind(node.left), bind(node.right))
        if isinstance(node, ast.UnaryOp):
            operand = bind(node.operand)
            return NotExpr(operand) if node.op == "NOT" else NegateExpr(operand)
        if isinstance(node, ast.IsNull):
            return IsNullExpr(bind(node.operand), node.negated)
        if isinstance(node, ast.FunctionCall):
            if not is_scalar_function(node.name):
                raise PlanError(f"unknown function {node.name}")
            return ScalarFunction(node.name, [bind(a) for a in node.args])
        raise PlanError(f"cannot plan expression node {type(node).__name__}")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _scan_relation(ref: ast.TableRef, table: Table, global_name: object) -> Operator:
    assert callable(global_name)
    scan = Scan(table, ref.bound_name)
    renames = [
        col(c).alias(global_name(ref.bound_name, c))
        for c in table.column_names
        if global_name(ref.bound_name, c) != c
    ]
    if not renames:
        return scan
    projections: list[Expression] = []
    for c in table.column_names:
        g = global_name(ref.bound_name, c)
        projections.append(col(c).alias(g) if g != c else col(c))
    return Project(scan, projections)


def _split_equijoin(
    predicate: Expression, left_names: set[str], right_names: set[str]
) -> tuple[list[Expression], list[Expression], Expression | None]:
    left_keys: list[Expression] = []
    right_keys: list[Expression] = []
    residual: list[Expression] = []
    for conjunct in _flatten_and(predicate):
        if isinstance(conjunct, BinaryExpr) and conjunct.op is BinOp.EQ:
            lrefs = conjunct.left.references()
            rrefs = conjunct.right.references()
            if lrefs <= left_names and rrefs <= right_names:
                left_keys.append(conjunct.left)
                right_keys.append(conjunct.right)
                continue
            if lrefs <= right_names and rrefs <= left_names:
                left_keys.append(conjunct.right)
                right_keys.append(conjunct.left)
                continue
        residual.append(conjunct)
    return left_keys, right_keys, _and_all(residual)


def _flatten_and(expr: Expression) -> list[Expression]:
    if isinstance(expr, BinaryExpr) and expr.op is BinOp.AND:
        return _flatten_and(expr.left) + _flatten_and(expr.right)
    return [expr]


def _and_all(terms: list[Expression]) -> Expression | None:
    if not terms:
        return None
    result = terms[0]
    for term in terms[1:]:
        result = BinaryExpr(BinOp.AND, result, term)
    return result


def _alias_expr(node: ast.Expr, alias_map: dict[str, ast.Expr]) -> ast.Expr:
    """Resolve a bare ORDER BY reference to a SELECT alias if one matches."""
    if isinstance(node, ast.ColumnRef) and node.table is None and node.name in alias_map:
        return alias_map[node.name]
    return node


def _output_name(node: ast.Expr) -> str:
    """A readable default output-column name derived from the SQL that wrote it."""
    if isinstance(node, ast.ColumnRef):
        return node.name
    if isinstance(node, ast.Literal):
        return "NULL" if node.value is None else repr(node.value)
    if isinstance(node, ast.FunctionCall):
        if node.star:
            return f"{node.name}(*)"
        inner = ", ".join(_output_name(a) for a in node.args)
        prefix = "DISTINCT " if node.distinct else ""
        return f"{node.name}({prefix}{inner})"
    if isinstance(node, ast.BinaryOp):
        return f"({_output_name(node.left)} {node.op} {_output_name(node.right)})"
    if isinstance(node, ast.UnaryOp):
        symbol = "NOT " if node.op == "NOT" else "-"
        return f"({symbol}{_output_name(node.operand)})"
    if isinstance(node, ast.IsNull):
        kind = "IS NOT NULL" if node.negated else "IS NULL"
        return f"({_output_name(node.operand)} {kind})"
    return "expr"


def _limit(op: Operator, stmt: ast.SelectStatement) -> Operator:
    if stmt.limit is not None or stmt.offset:
        return Limit(op, stmt.limit, stmt.offset)
    return op


def _unit_table() -> Table:
    """A single-row, single-column table for ``SELECT`` with no ``FROM``."""
    return Table([Column.from_pylist("__unit", [0], DataType.INTEGER)])


def _has_aggregate(stmt: ast.SelectStatement) -> bool:
    if stmt.having is not None:
        return True
    if any(_contains_aggregate(item.expr) for item in stmt.items):
        return True
    return any(_contains_aggregate(ok.expr) for ok in stmt.order_by)


def _contains_aggregate(node: ast.Expr | None) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.FunctionCall):
        return is_aggregate(node.name) or any(_contains_aggregate(a) for a in node.args)
    if isinstance(node, ast.BinaryOp):
        return _contains_aggregate(node.left) or _contains_aggregate(node.right)
    if isinstance(node, ast.UnaryOp):
        return _contains_aggregate(node.operand)
    if isinstance(node, ast.IsNull):
        return _contains_aggregate(node.operand)
    return False
