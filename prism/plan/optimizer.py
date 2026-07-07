"""A rule-based query optimizer over the physical operator tree.

prism unifies the logical and physical plan into one operator tree, so the
optimizer rewrites that tree directly, applying semantics-preserving rules until
the plan stops changing:

* **Constant folding** evaluates expressions with no column inputs
  (``1 + 2`` becomes ``3``) and simplifies boolean identities
  (``x AND TRUE`` becomes ``x``, ``x OR TRUE`` becomes ``TRUE``).
* **Predicate pushdown** moves filters down the tree toward the scans: through
  projections (substituting aliases), through sorts, and split across joins so
  each conjunct runs on the side that owns its columns. Filtering early shrinks
  every intermediate result above it.
* **Projection pushdown / column pruning** works out the columns each subtree
  actually needs and inserts a narrow projection at each scan, so a columnar
  read never touches columns the query ignores.

Every rule returns a new tree; nothing is mutated in place. The rewrite loop
runs to a fixpoint so that, for example, a filter pushed into a join can be
pushed again into that join's inputs on the next pass.
"""

from __future__ import annotations

from prism.aggregate import AggregateExpr
from prism.column import Column
from prism.exec.aggregate import Distinct, HashAggregate
from prism.exec.join import HashJoin, NestedLoopJoin
from prism.exec.operators import Filter, Limit, Operator, Project, Scan, Sort, SortKey
from prism.expr import (
    AliasExpr,
    BinaryExpr,
    BinOp,
    ColumnRef,
    Expression,
    IsNullExpr,
    Literal,
    NegateExpr,
    NotExpr,
    ScalarFunction,
    col,
)
from prism.table import Table
from prism.types import DataType


def optimize(op: Operator) -> Operator:
    """Return an equivalent operator tree with the rewrite rules applied."""
    op = _fold_all(op)
    while True:
        op, changed = _rewrite(op)
        if not changed:
            break
    return _prune_columns(op, set(op.schema()))


# ----------------------------------------------------------------------
# expression rewriting
# ----------------------------------------------------------------------


def _expr_children(e: Expression) -> list[Expression]:
    if isinstance(e, AliasExpr):
        return [e.inner]
    if isinstance(e, BinaryExpr):
        return [e.left, e.right]
    if isinstance(e, NotExpr | NegateExpr):
        return [e.operand]
    if isinstance(e, IsNullExpr):
        return [e.operand]
    if isinstance(e, ScalarFunction):
        return list(e.args)
    if isinstance(e, AggregateExpr):
        return [e.arg] if e.arg is not None else []
    return []


def _rebuild_expr(e: Expression, children: list[Expression]) -> Expression:
    if isinstance(e, AliasExpr):
        return AliasExpr(children[0], e.name)
    if isinstance(e, BinaryExpr):
        return BinaryExpr(e.op, children[0], children[1])
    if isinstance(e, NotExpr):
        return NotExpr(children[0])
    if isinstance(e, NegateExpr):
        return NegateExpr(children[0])
    if isinstance(e, IsNullExpr):
        return IsNullExpr(children[0], e.negated)
    if isinstance(e, ScalarFunction):
        return ScalarFunction(e.name, children)
    if isinstance(e, AggregateExpr):
        return AggregateExpr(e.func, children[0] if children else None, e.distinct)
    return e


def _contains_aggregate(e: Expression) -> bool:
    if isinstance(e, AggregateExpr):
        return True
    return any(_contains_aggregate(c) for c in _expr_children(e))


def _fold_expr(e: Expression) -> Expression:
    """Constant-fold and simplify an expression bottom up."""
    node = _rebuild_expr(e, [_fold_expr(c) for c in _expr_children(e)])
    node = _simplify_boolean(node)
    # A column-free, aggregate-free expression is a constant we can evaluate now.
    if (
        not isinstance(node, Literal | AliasExpr | ColumnRef)
        and not node.references()
        and not _contains_aggregate(node)
    ):
        return _to_literal(node)
    return node


def _simplify_boolean(node: Expression) -> Expression:
    if not (isinstance(node, BinaryExpr) and node.op in (BinOp.AND, BinOp.OR)):
        return node
    left_val = _bool_literal(node.left)
    right_val = _bool_literal(node.right)
    if node.op is BinOp.AND:
        if left_val is True:
            return node.right
        if right_val is True:
            return node.left
        if left_val is False or right_val is False:
            return Literal(False, DataType.BOOLEAN)
    else:  # OR
        if left_val is False:
            return node.right
        if right_val is False:
            return node.left
        if left_val is True or right_val is True:
            return Literal(True, DataType.BOOLEAN)
    return node


def _bool_literal(e: Expression) -> bool | None:
    if isinstance(e, Literal) and e.dtype is DataType.BOOLEAN:
        return bool(e.value)
    return None


_UNIT = Table([Column.from_pylist("__unit", [0], DataType.INTEGER)])


def _to_literal(node: Expression) -> Literal:
    column = node.evaluate(_UNIT)
    return Literal(column.to_pylist()[0], column.dtype)


def _substitute(e: Expression, mapping: dict[str, Expression]) -> Expression:
    if isinstance(e, ColumnRef) and e.name in mapping:
        return mapping[e.name]
    return _rebuild_expr(e, [_substitute(c, mapping) for c in _expr_children(e)])


# ----------------------------------------------------------------------
# operator rebuilding
# ----------------------------------------------------------------------


def _rebuild(op: Operator, children: list[Operator]) -> Operator:
    if isinstance(op, Filter):
        return Filter(children[0], op.predicate)
    if isinstance(op, Project):
        return Project(children[0], op.projections)
    if isinstance(op, Sort):
        return Sort(children[0], op.keys)
    if isinstance(op, Limit):
        return Limit(children[0], op.limit, op.offset)
    if isinstance(op, Distinct):
        return Distinct(children[0])
    if isinstance(op, HashAggregate):
        return HashAggregate(children[0], op.group_exprs, op.specs)
    if isinstance(op, HashJoin):
        return HashJoin(children[0], children[1], op.left_keys, op.right_keys, op.how, op.residual)
    if isinstance(op, NestedLoopJoin):
        return NestedLoopJoin(children[0], children[1], op.predicate, op.how)
    return op


def _fold_all(op: Operator) -> Operator:
    if op.children:
        op = _rebuild(op, [_fold_all(c) for c in op.children])
    return _fold_operator_exprs(op)


def _fold_operator_exprs(op: Operator) -> Operator:
    if isinstance(op, Filter):
        return Filter(op.child, _fold_expr(op.predicate))
    if isinstance(op, Project):
        return Project(op.child, [_fold_expr(p) for p in op.projections])
    if isinstance(op, Sort):
        keys = [SortKey(_fold_expr(k.expression), k.ascending, k.nulls_first) for k in op.keys]
        return Sort(op.child, keys)
    if isinstance(op, HashAggregate):
        return HashAggregate(op.child, [_fold_expr(e) for e in op.group_exprs], op.specs)
    if isinstance(op, HashJoin):
        residual = _fold_expr(op.residual) if op.residual is not None else None
        return HashJoin(
            op.left,
            op.right,
            [_fold_expr(k) for k in op.left_keys],
            [_fold_expr(k) for k in op.right_keys],
            op.how,
            residual,
        )
    if isinstance(op, NestedLoopJoin):
        predicate = _fold_expr(op.predicate) if op.predicate is not None else None
        return NestedLoopJoin(op.left, op.right, predicate, op.how)
    return op


# ----------------------------------------------------------------------
# predicate pushdown
# ----------------------------------------------------------------------


def _rewrite(op: Operator) -> tuple[Operator, bool]:
    changed = False
    if op.children:
        new_children = []
        for child in op.children:
            rewritten, child_changed = _rewrite(child)
            new_children.append(rewritten)
            changed = changed or child_changed
        op = _rebuild(op, new_children)

    if isinstance(op, Filter):
        if _is_true(op.predicate):
            return op.child, True
        pushed = _push_filter(op)
        if pushed is not None:
            return pushed, True
    return op, changed


def _push_filter(node: Filter) -> Operator | None:
    """Try to move ``node`` closer to the scans. Returns None if it cannot."""
    child = node.child
    predicate = node.predicate

    if isinstance(child, Filter):  # merge stacked filters into one conjunction
        return Filter(child.child, _and(child.predicate, predicate))

    if isinstance(child, Project):
        mapping = {p.output_name(): _unalias(p) for p in child.projections}
        substituted = _substitute(predicate, mapping)
        available = set(child.child.schema())
        if not _contains_aggregate(substituted) and substituted.references() <= available:
            return Project(Filter(child.child, substituted), child.projections)

    if isinstance(child, Sort):  # filtering commutes with ordering
        return Sort(Filter(child.child, predicate), child.keys)

    if isinstance(child, HashJoin | NestedLoopJoin):
        return _push_into_join(child, predicate)

    return None


def _push_into_join(join: HashJoin | NestedLoopJoin, predicate: Expression) -> Operator | None:
    left_cols = set(join.left.schema())
    right_cols = set(join.right.schema())
    push_left: list[Expression] = []
    push_right: list[Expression] = []
    keep: list[Expression] = []

    for conjunct in _split_and(predicate):
        refs = conjunct.references()
        if refs <= left_cols:
            push_left.append(conjunct)  # safe for INNER and the preserved LEFT side
        elif refs <= right_cols and join.how == "INNER":
            push_right.append(conjunct)
        else:
            keep.append(conjunct)

    if not push_left and not push_right:
        return None

    new_left = Filter(join.left, _and_all(push_left)) if push_left else join.left
    new_right = Filter(join.right, _and_all(push_right)) if push_right else join.right
    new_join = _rebuild(join, [new_left, new_right])
    return Filter(new_join, _and_all(keep)) if keep else new_join


# ----------------------------------------------------------------------
# column pruning
# ----------------------------------------------------------------------


def _prune_columns(op: Operator, required: set[str]) -> Operator:
    child_req: set[str]
    if isinstance(op, Scan):
        keep = [c for c in op.table.column_names if c in required]
        if keep and len(keep) < op.table.num_columns:
            return Project(op, [col(c) for c in keep])
        return op

    if isinstance(op, Filter):
        child_req = required | op.predicate.references()
        return Filter(_prune_columns(op.child, child_req), op.predicate)

    if isinstance(op, Project):
        child_req = set()
        for p in op.projections:
            child_req |= p.references()
        return Project(_prune_columns(op.child, child_req), op.projections)

    if isinstance(op, Sort):
        child_req = set(required)
        for key in op.keys:
            child_req |= key.expression.references()
        return Sort(_prune_columns(op.child, child_req), op.keys)

    if isinstance(op, Limit):
        return Limit(_prune_columns(op.child, required), op.limit, op.offset)

    if isinstance(op, Distinct):
        return Distinct(_prune_columns(op.child, required))

    if isinstance(op, HashAggregate):
        child_req = set()
        for e in op.group_exprs:
            child_req |= e.references()
        for spec in op.specs:
            if spec.arg is not None:
                child_req |= spec.arg.references()
        return HashAggregate(_prune_columns(op.child, child_req), op.group_exprs, op.specs)

    if isinstance(op, HashJoin):
        left_req, right_req = _join_child_requirements(
            op, required, left_extra=_refs(op.left_keys), right_extra=_refs(op.right_keys)
        )
        return HashJoin(
            _prune_columns(op.left, left_req),
            _prune_columns(op.right, right_req),
            op.left_keys,
            op.right_keys,
            op.how,
            op.residual,
        )

    if isinstance(op, NestedLoopJoin):
        left_req, right_req = _join_child_requirements(op, required)
        return NestedLoopJoin(
            _prune_columns(op.left, left_req),
            _prune_columns(op.right, right_req),
            op.predicate,
            op.how,
        )

    return op


def _join_child_requirements(
    join: HashJoin | NestedLoopJoin,
    required: set[str],
    left_extra: set[str] | None = None,
    right_extra: set[str] | None = None,
) -> tuple[set[str], set[str]]:
    left_cols = set(join.left.schema())
    right_cols = set(join.right.schema())
    predicate = join.predicate if isinstance(join, NestedLoopJoin) else join.residual
    pred_refs = predicate.references() if predicate is not None else set()
    left_req = (required & left_cols) | (pred_refs & left_cols) | (left_extra or set())
    right_req = (required & right_cols) | (pred_refs & right_cols) | (right_extra or set())
    return left_req, right_req


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------


def _unalias(e: Expression) -> Expression:
    return e.inner if isinstance(e, AliasExpr) else e


def _refs(exprs: list[Expression]) -> set[str]:
    out: set[str] = set()
    for e in exprs:
        out |= e.references()
    return out


def _is_true(e: Expression) -> bool:
    return isinstance(e, Literal) and e.dtype is DataType.BOOLEAN and bool(e.value) is True


def _split_and(expr: Expression) -> list[Expression]:
    if isinstance(expr, BinaryExpr) and expr.op is BinOp.AND:
        return _split_and(expr.left) + _split_and(expr.right)
    return [expr]


def _and(left: Expression, right: Expression) -> Expression:
    return BinaryExpr(BinOp.AND, left, right)


def _and_all(terms: list[Expression]) -> Expression:
    result = terms[0]
    for term in terms[1:]:
        result = _and(result, term)
    return result
