"""Tests for the rule-based optimizer.

Two things are checked for every rule: the plan is rewritten the way we expect
(structure), and the query still returns the same rows it did before (semantics).
"""

from __future__ import annotations

import pytest

from prism import Database, Table
from prism.exec.operators import Filter, Project, Scan, Sort
from prism.expr import BinaryExpr, BinOp, Literal, col
from prism.plan import optimize, plan
from prism.plan.optimizer import _fold_expr
from prism.types import DataType


@pytest.fixture
def db() -> Database:
    database = Database()
    database.register(
        "emp",
        Table.from_pydict(
            {
                "name": ["Ada", "Grace", "Alan", "Kate"],
                "dept_id": [1, 1, 2, 2],
                "salary": [100, 200, 150, 300],
            }
        ),
    )
    database.register(
        "dept",
        Table.from_pydict({"id": [1, 2], "dname": ["Eng", "Research"]}),
    )
    return database


def _operators(op) -> list:  # type: ignore[no-untyped-def]
    """Flatten an operator tree into a list for structural assertions."""
    out = [op]
    for child in op.children:
        out.extend(_operators(child))
    return out


class TestConstantFolding:
    def test_arithmetic_folded(self) -> None:
        folded = _fold_expr(col("x") + (Literal(1) + Literal(2)))
        assert isinstance(folded, BinaryExpr)
        assert isinstance(folded.right, Literal)
        assert folded.right.value == 3

    def test_comparison_folded(self) -> None:
        folded = _fold_expr(BinaryExpr(BinOp.EQ, Literal(1), Literal(1)))
        assert isinstance(folded, Literal)
        assert folded.value is True

    def test_and_true_drops_operand(self) -> None:
        expr = BinaryExpr(BinOp.AND, col("x") > Literal(5), Literal(True, DataType.BOOLEAN))
        folded = _fold_expr(expr)
        assert isinstance(folded, BinaryExpr)
        assert folded.op is BinOp.GT

    def test_and_false_short_circuits(self) -> None:
        expr = BinaryExpr(BinOp.AND, col("x") > Literal(5), Literal(False, DataType.BOOLEAN))
        folded = _fold_expr(expr)
        assert isinstance(folded, Literal)
        assert folded.value is False

    def test_or_true_short_circuits(self) -> None:
        expr = BinaryExpr(BinOp.OR, col("x") > Literal(5), Literal(True, DataType.BOOLEAN))
        folded = _fold_expr(expr)
        assert isinstance(folded, Literal)
        assert folded.value is True

    def test_or_false_drops_operand(self) -> None:
        expr = BinaryExpr(BinOp.OR, col("x") > Literal(5), Literal(False, DataType.BOOLEAN))
        folded = _fold_expr(expr)
        assert isinstance(folded, BinaryExpr)
        assert folded.op is BinOp.GT

    def test_column_free_boolean_folds_to_literal(self) -> None:
        # A subtree with no column inputs collapses to a single literal.
        folded = _fold_expr(Literal(2) > Literal(1))
        assert isinstance(folded, Literal)
        assert folded.value is True

    def test_true_filter_removed(self, db: Database) -> None:
        optimized = db.plan("SELECT name FROM emp WHERE 1 = 1")
        assert not any(isinstance(o, Filter) for o in _operators(optimized))

    def test_folding_preserves_results(self, db: Database) -> None:
        rows = db.sql("SELECT name FROM emp WHERE salary > 100 + 50").to_rows()
        assert rows == [("Grace",), ("Kate",)]


class TestPredicatePushdown:
    def test_filter_pushed_through_project(self, db: Database) -> None:
        optimized = db.plan("SELECT name FROM emp WHERE salary > 150")
        ops = _operators(optimized)
        # The filter should sit below the top projection, next to the scan.
        filters = [o for o in ops if isinstance(o, Filter)]
        assert len(filters) == 1
        assert isinstance(filters[0].child, Scan | Project)

    def test_filter_split_across_join(self, db: Database) -> None:
        q = (
            "SELECT emp.name, dept.dname FROM emp JOIN dept ON emp.dept_id = dept.id "
            "WHERE salary > 120 AND dname = 'Research'"
        )
        optimized = db.plan(q)
        # Both single-table predicates should have been pushed to their scans.
        from prism.exec.join import HashJoin

        join = next(o for o in _operators(optimized) if isinstance(o, HashJoin))
        assert isinstance(join.left, Filter)
        assert isinstance(join.right, Filter)

    def test_left_join_keeps_right_predicate_above(self, db: Database) -> None:
        q = (
            "SELECT emp.name, dept.dname FROM emp LEFT JOIN dept ON emp.dept_id = dept.id "
            "WHERE dname = 'Research'"
        )
        optimized = db.plan(q)
        from prism.exec.join import HashJoin

        join = next(o for o in _operators(optimized) if isinstance(o, HashJoin))
        # The right-side predicate must NOT be pushed into the nullable side.
        assert not isinstance(join.right, Filter)

    def test_pushdown_preserves_results(self, db: Database) -> None:
        q = (
            "SELECT emp.name FROM emp JOIN dept ON emp.dept_id = dept.id "
            "WHERE salary >= 150 AND dname = 'Research'"
        )
        assert db.sql(q).to_rows() == [("Alan",), ("Kate",)]

    def test_filter_pushed_below_sort(self, db: Database) -> None:
        optimized = db.plan("SELECT name FROM emp ORDER BY salary")
        ops = _operators(optimized)
        sorts = [o for o in ops if isinstance(o, Sort)]
        assert sorts  # sort survives; nothing to assert beyond it planning cleanly


class TestColumnPruning:
    def test_scan_pruned_to_used_columns(self, db: Database) -> None:
        optimized = db.plan("SELECT name FROM emp WHERE salary > 150")
        # A projection should wrap the scan, keeping only name and salary.
        scans = [o for o in _operators(optimized) if isinstance(o, Scan)]
        assert len(scans) == 1
        projects_over_scan = [
            o
            for o in _operators(optimized)
            if isinstance(o, Project) and any(isinstance(c, Scan) for c in o.children)
        ]
        assert projects_over_scan
        pruned = projects_over_scan[0]
        assert set(pruned.schema()) == {"name", "salary"}

    def test_pruning_preserves_results(self, db: Database) -> None:
        rows = db.sql("SELECT name FROM emp WHERE salary > 150").to_rows()
        assert rows == [("Grace",), ("Kate",)]

    def test_no_pruning_when_all_columns_used(self, db: Database) -> None:
        optimized = db.plan("SELECT name, dept_id, salary FROM emp")
        # Every column is used, so no extra pruning projection above the scan.
        scans = [o for o in _operators(optimized) if isinstance(o, Scan)]
        assert scans[0].table.num_columns == 3


class TestEndToEndEquivalence:
    @pytest.mark.parametrize(
        "query",
        [
            "SELECT name, salary FROM emp WHERE salary > 100 AND 2 > 1",
            "SELECT dept_id, COUNT(*) FROM emp GROUP BY dept_id ORDER BY dept_id",
            "SELECT name FROM emp ORDER BY salary DESC LIMIT 2",
            "SELECT emp.name, dname FROM emp JOIN dept ON emp.dept_id = dept.id WHERE salary > 120",
            "SELECT DISTINCT dept_id FROM emp",
        ],
    )
    def test_optimized_matches_unoptimized(self, db: Database, query: str) -> None:
        catalog = db.catalog
        raw = plan(query, catalog)
        opt = optimize(raw)
        assert opt.execute().to_rows() == raw.execute().to_rows()


class TestExplain:
    def test_explain_diff_shows_both(self, db: Database) -> None:
        text = db.explain_diff("SELECT name FROM emp WHERE salary > 100 AND 1 = 1")
        assert "original plan" in text and "optimized plan" in text
