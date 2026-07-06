"""A hand-written SQL parser: recursive descent, with precedence for expressions.

The grammar covers a practical ``SELECT``: projection list with aliases and
``*``, ``FROM`` with comma tables and ``INNER``/``LEFT`` joins, ``WHERE``,
``GROUP BY``, ``HAVING``, ``ORDER BY`` with direction and null placement, and
``LIMIT``/``OFFSET``. Expressions are parsed by precedence climbing so that
``a OR b AND c = d + e`` associates the way SQL requires.
"""

from __future__ import annotations

from prism.sql import ast
from prism.sql.lexer import tokenize
from prism.sql.tokens import Token, TokenType
from prism.types import DataType

# Comparison tokens and the operator string the AST records for each.
_COMPARISONS = {
    TokenType.EQ: "=",
    TokenType.NEQ: "<>",
    TokenType.LT: "<",
    TokenType.LTE: "<=",
    TokenType.GT: ">",
    TokenType.GTE: ">=",
}
_ADDITIVE = {TokenType.PLUS: "+", TokenType.MINUS: "-"}
_MULTIPLICATIVE = {TokenType.STAR: "*", TokenType.SLASH: "/", TokenType.PERCENT: "%"}


class ParseError(ValueError):
    """Raised when the token stream does not form a valid query."""


def parse(sql: str) -> ast.SelectStatement:
    """Parse a single SELECT statement from ``sql``."""
    return _Parser(tokenize(sql)).parse_statement()


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    # -- token cursor ---------------------------------------------------

    def _peek(self, ahead: int = 0) -> Token:
        index = min(self.pos + ahead, len(self.tokens) - 1)
        return self.tokens[index]

    def _advance(self) -> Token:
        token = self.tokens[self.pos]
        if token.type is not TokenType.EOF:
            self.pos += 1
        return token

    def _check(self, ttype: TokenType) -> bool:
        return self._peek().type is ttype

    def _match(self, ttype: TokenType) -> bool:
        if self._check(ttype):
            self._advance()
            return True
        return False

    def _expect(self, ttype: TokenType) -> Token:
        if not self._check(ttype):
            got = self._peek()
            raise ParseError(f"expected {ttype.name}, got {got.type.name} {got.text!r}")
        return self._advance()

    def _check_kw(self, keyword: str) -> bool:
        tok = self._peek()
        return tok.type is TokenType.KEYWORD and tok.value == keyword

    def _match_kw(self, keyword: str) -> bool:
        if self._check_kw(keyword):
            self._advance()
            return True
        return False

    def _expect_kw(self, keyword: str) -> Token:
        if not self._check_kw(keyword):
            got = self._peek()
            raise ParseError(f"expected keyword {keyword}, got {got.text!r}")
        return self._advance()

    # -- statement ------------------------------------------------------

    def parse_statement(self) -> ast.SelectStatement:
        self._expect_kw("SELECT")
        distinct = self._match_kw("DISTINCT")
        items = self._parse_select_items()

        from_table: ast.TableRef | None = None
        extra: list[ast.TableRef] = []
        joins: list[ast.Join] = []
        if self._match_kw("FROM"):
            from_table = self._parse_table_ref()
            while self._match(TokenType.COMMA):
                extra.append(self._parse_table_ref())
            while self._is_join_start():
                joins.append(self._parse_join())

        where = self._parse_expr() if self._match_kw("WHERE") else None

        group_by: list[ast.Expr] = []
        if self._match_kw("GROUP"):
            self._expect_kw("BY")
            group_by.append(self._parse_expr())
            while self._match(TokenType.COMMA):
                group_by.append(self._parse_expr())

        having = self._parse_expr() if self._match_kw("HAVING") else None

        order_by: list[ast.OrderKey] = []
        if self._match_kw("ORDER"):
            self._expect_kw("BY")
            order_by.append(self._parse_order_key())
            while self._match(TokenType.COMMA):
                order_by.append(self._parse_order_key())

        limit: int | None = None
        offset = 0
        if self._match_kw("LIMIT"):
            limit = self._parse_int_literal("LIMIT")
        if self._match_kw("OFFSET"):
            offset = self._parse_int_literal("OFFSET")

        self._match(TokenType.SEMICOLON)
        if not self._check(TokenType.EOF):
            got = self._peek()
            raise ParseError(f"unexpected trailing input {got.text!r}")

        return ast.SelectStatement(
            items=tuple(items),
            from_table=from_table,
            joins=tuple(joins),
            where=where,
            group_by=tuple(group_by),
            having=having,
            order_by=tuple(order_by),
            limit=limit,
            offset=offset,
            distinct=distinct,
            extra_tables=tuple(extra),
        )

    def _parse_select_items(self) -> list[ast.SelectItem]:
        items = [self._parse_select_item()]
        while self._match(TokenType.COMMA):
            items.append(self._parse_select_item())
        return items

    def _parse_select_item(self) -> ast.SelectItem:
        # Bare "*"
        if self._check(TokenType.STAR):
            self._advance()
            return ast.SelectItem(ast.Star(None))
        # Qualified "t.*"
        if (
            self._check(TokenType.IDENT)
            and self._peek(1).type is TokenType.DOT
            and self._peek(2).type is TokenType.STAR
        ):
            table = self._advance().text
            self._advance()  # DOT
            self._advance()  # STAR
            return ast.SelectItem(ast.Star(table))

        expr = self._parse_expr()
        alias: str | None = None
        if self._match_kw("AS"):
            alias = self._expect(TokenType.IDENT).text
        elif self._check(TokenType.IDENT):
            alias = self._advance().text
        return ast.SelectItem(expr, alias)

    def _parse_table_ref(self) -> ast.TableRef:
        name = self._expect(TokenType.IDENT).text
        alias: str | None = None
        if self._match_kw("AS"):
            alias = self._expect(TokenType.IDENT).text
        elif self._check(TokenType.IDENT):
            alias = self._advance().text
        return ast.TableRef(name, alias)

    def _is_join_start(self) -> bool:
        return self._check_kw("JOIN") or self._check_kw("INNER") or self._check_kw("LEFT")

    def _parse_join(self) -> ast.Join:
        kind = "INNER"
        if self._match_kw("INNER"):
            self._expect_kw("JOIN")
        elif self._match_kw("LEFT"):
            self._match_kw("OUTER")
            self._expect_kw("JOIN")
            kind = "LEFT"
        else:
            self._expect_kw("JOIN")
        right = self._parse_table_ref()
        self._expect_kw("ON")
        on = self._parse_expr()
        return ast.Join(kind, right, on)

    def _parse_order_key(self) -> ast.OrderKey:
        expr = self._parse_expr()
        ascending = True
        if self._match_kw("ASC"):
            ascending = True
        elif self._match_kw("DESC"):
            ascending = False
        nulls_first: bool | None = None
        if self._match_kw("NULLS"):
            if self._match_kw("FIRST"):
                nulls_first = True
            elif self._match_kw("LAST"):
                nulls_first = False
            else:
                raise ParseError("expected FIRST or LAST after NULLS")
        return ast.OrderKey(expr, ascending, nulls_first)

    def _parse_int_literal(self, clause: str) -> int:
        tok = self._expect(TokenType.NUMBER)
        if not isinstance(tok.value, int):
            raise ParseError(f"{clause} requires an integer, got {tok.text!r}")
        return tok.value

    # -- expressions (precedence climbing) ------------------------------

    def _parse_expr(self) -> ast.Expr:
        return self._parse_or()

    def _parse_or(self) -> ast.Expr:
        left = self._parse_and()
        while self._match_kw("OR"):
            left = ast.BinaryOp("OR", left, self._parse_and())
        return left

    def _parse_and(self) -> ast.Expr:
        left = self._parse_not()
        while self._match_kw("AND"):
            left = ast.BinaryOp("AND", left, self._parse_not())
        return left

    def _parse_not(self) -> ast.Expr:
        if self._match_kw("NOT"):
            return ast.UnaryOp("NOT", self._parse_not())
        return self._parse_comparison()

    def _parse_comparison(self) -> ast.Expr:
        left = self._parse_additive()
        if self._match_kw("IS"):
            negated = self._match_kw("NOT")
            self._expect_kw("NULL")
            return ast.IsNull(left, negated)
        op = _COMPARISONS.get(self._peek().type)
        if op is not None:
            self._advance()
            return ast.BinaryOp(op, left, self._parse_additive())
        return left

    def _parse_additive(self) -> ast.Expr:
        left = self._parse_multiplicative()
        while (op := _ADDITIVE.get(self._peek().type)) is not None:
            self._advance()
            left = ast.BinaryOp(op, left, self._parse_multiplicative())
        return left

    def _parse_multiplicative(self) -> ast.Expr:
        left = self._parse_unary()
        while (op := _MULTIPLICATIVE.get(self._peek().type)) is not None:
            self._advance()
            left = ast.BinaryOp(op, left, self._parse_unary())
        return left

    def _parse_unary(self) -> ast.Expr:
        if self._match(TokenType.MINUS):
            return ast.UnaryOp("-", self._parse_unary())
        if self._match(TokenType.PLUS):
            return self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> ast.Expr:
        tok = self._peek()

        if tok.type is TokenType.NUMBER:
            self._advance()
            dtype = DataType.INTEGER if isinstance(tok.value, int) else DataType.FLOAT
            return ast.Literal(tok.value, dtype)

        if tok.type is TokenType.STRING:
            self._advance()
            return ast.Literal(tok.value, DataType.TEXT)

        if tok.type is TokenType.KEYWORD:
            if tok.value in ("TRUE", "FALSE"):
                self._advance()
                return ast.Literal(tok.value == "TRUE", DataType.BOOLEAN)
            if tok.value == "NULL":
                self._advance()
                return ast.Literal(None, DataType.NULL)
            raise ParseError(f"unexpected keyword {tok.text!r} in expression")

        if self._match(TokenType.LPAREN):
            inner = self._parse_expr()
            self._expect(TokenType.RPAREN)
            return inner

        if tok.type is TokenType.IDENT:
            return self._parse_ident_expr()

        raise ParseError(f"unexpected token {tok.text!r} in expression")

    def _parse_ident_expr(self) -> ast.Expr:
        name = self._advance().text

        if self._check(TokenType.LPAREN):
            return self._parse_function_call(name)

        if self._match(TokenType.DOT):
            column = self._expect(TokenType.IDENT).text
            return ast.ColumnRef(column, table=name)

        return ast.ColumnRef(name)

    def _parse_function_call(self, name: str) -> ast.FunctionCall:
        self._expect(TokenType.LPAREN)
        if self._match(TokenType.STAR):
            self._expect(TokenType.RPAREN)
            return ast.FunctionCall(name.upper(), star=True)

        distinct = self._match_kw("DISTINCT")
        args: list[ast.Expr] = []
        if not self._check(TokenType.RPAREN):
            args.append(self._parse_expr())
            while self._match(TokenType.COMMA):
                args.append(self._parse_expr())
        self._expect(TokenType.RPAREN)
        return ast.FunctionCall(name.upper(), tuple(args), distinct=distinct)
