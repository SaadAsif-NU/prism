"""Tests for the SQL parser and the AST it builds."""

from __future__ import annotations

import pytest

from prism.sql import ast, parse
from prism.sql.parser import ParseError


class TestSelectList:
    def test_star(self) -> None:
        stmt = parse("SELECT * FROM t")
        assert isinstance(stmt.items[0].expr, ast.Star)

    def test_qualified_star(self) -> None:
        stmt = parse("SELECT t.* FROM t")
        assert stmt.items[0].expr == ast.Star("t")

    def test_columns_with_aliases(self) -> None:
        stmt = parse("SELECT a, b AS bee, c cee FROM t")
        assert stmt.items[0].alias is None
        assert stmt.items[1].alias == "bee"
        assert stmt.items[2].alias == "cee"

    def test_distinct(self) -> None:
        assert parse("SELECT DISTINCT a FROM t").distinct


class TestExpressions:
    def test_precedence_or_and(self) -> None:
        stmt = parse("SELECT * FROM t WHERE a OR b AND c")
        # a OR (b AND c)
        assert isinstance(stmt.where, ast.BinaryOp)
        assert stmt.where.op == "OR"
        assert stmt.where.right.op == "AND"

    def test_arithmetic_precedence(self) -> None:
        stmt = parse("SELECT a + b * c FROM t")
        expr = stmt.items[0].expr
        assert expr.op == "+"
        assert expr.right.op == "*"

    def test_comparison(self) -> None:
        stmt = parse("SELECT * FROM t WHERE a >= 10")
        assert stmt.where.op == ">="

    def test_is_null(self) -> None:
        stmt = parse("SELECT * FROM t WHERE a IS NULL")
        assert isinstance(stmt.where, ast.IsNull)
        assert not stmt.where.negated

    def test_is_not_null(self) -> None:
        stmt = parse("SELECT * FROM t WHERE a IS NOT NULL")
        assert stmt.where.negated

    def test_not(self) -> None:
        stmt = parse("SELECT * FROM t WHERE NOT a")
        assert isinstance(stmt.where, ast.UnaryOp)
        assert stmt.where.op == "NOT"

    def test_unary_minus(self) -> None:
        stmt = parse("SELECT -a FROM t")
        assert stmt.items[0].expr.op == "-"

    def test_parentheses(self) -> None:
        stmt = parse("SELECT (a + b) * c FROM t")
        assert stmt.items[0].expr.op == "*"
        assert stmt.items[0].expr.left.op == "+"

    def test_qualified_column(self) -> None:
        stmt = parse("SELECT t.a FROM t")
        assert stmt.items[0].expr == ast.ColumnRef("a", "t")

    def test_literals(self) -> None:
        stmt = parse("SELECT 1, 2.5, 'x', TRUE, NULL")
        values = [i.expr.value for i in stmt.items]
        assert values == [1, 2.5, "x", True, None]


class TestFunctions:
    def test_count_star(self) -> None:
        stmt = parse("SELECT COUNT(*) FROM t")
        call = stmt.items[0].expr
        assert call.name == "COUNT" and call.star

    def test_aggregate_arg(self) -> None:
        stmt = parse("SELECT SUM(salary) FROM t")
        assert stmt.items[0].expr.name == "SUM"

    def test_distinct_aggregate(self) -> None:
        stmt = parse("SELECT COUNT(DISTINCT dept) FROM t")
        assert stmt.items[0].expr.distinct

    def test_scalar_function(self) -> None:
        stmt = parse("SELECT ROUND(x, 2) FROM t")
        assert stmt.items[0].expr.name == "ROUND"
        assert len(stmt.items[0].expr.args) == 2


class TestClauses:
    def test_from_alias(self) -> None:
        stmt = parse("SELECT * FROM employees e")
        assert stmt.from_table == ast.TableRef("employees", "e")

    def test_inner_join(self) -> None:
        stmt = parse("SELECT * FROM a JOIN b ON a.id = b.id")
        assert stmt.joins[0].kind == "INNER"

    def test_left_join(self) -> None:
        stmt = parse("SELECT * FROM a LEFT OUTER JOIN b ON a.id = b.id")
        assert stmt.joins[0].kind == "LEFT"

    def test_comma_tables(self) -> None:
        stmt = parse("SELECT * FROM a, b")
        assert len(stmt.extra_tables) == 1

    def test_group_by_having(self) -> None:
        stmt = parse("SELECT a FROM t GROUP BY a HAVING COUNT(*) > 1")
        assert len(stmt.group_by) == 1
        assert stmt.having is not None

    def test_order_by_directions(self) -> None:
        stmt = parse("SELECT * FROM t ORDER BY a ASC, b DESC")
        assert stmt.order_by[0].ascending
        assert not stmt.order_by[1].ascending

    def test_order_by_nulls(self) -> None:
        stmt = parse("SELECT * FROM t ORDER BY a NULLS FIRST")
        assert stmt.order_by[0].nulls_first is True

    def test_limit_offset(self) -> None:
        stmt = parse("SELECT * FROM t LIMIT 10 OFFSET 5")
        assert stmt.limit == 10 and stmt.offset == 5

    def test_trailing_semicolon(self) -> None:
        assert parse("SELECT 1;").items  # does not raise


class TestErrors:
    def test_missing_select(self) -> None:
        with pytest.raises(ParseError):
            parse("FROM t")

    def test_trailing_garbage(self) -> None:
        with pytest.raises(ParseError, match="trailing"):
            parse("SELECT 1 garbage extra")

    def test_unclosed_paren(self) -> None:
        with pytest.raises(ParseError):
            parse("SELECT (a + b FROM t")

    def test_limit_requires_integer(self) -> None:
        with pytest.raises(ParseError, match="integer"):
            parse("SELECT * FROM t LIMIT 1.5")

    def test_bad_nulls_clause(self) -> None:
        with pytest.raises(ParseError, match="FIRST or LAST"):
            parse("SELECT * FROM t ORDER BY a NULLS SOON")
