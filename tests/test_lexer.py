"""Tests for the SQL lexer."""

from __future__ import annotations

import pytest

from prism.sql.lexer import LexError, tokenize
from prism.sql.tokens import TokenType


def _types(sql: str) -> list[TokenType]:
    return [t.type for t in tokenize(sql)]


class TestScanning:
    def test_keywords_and_identifiers(self) -> None:
        toks = tokenize("SELECT name FROM t")
        assert toks[0].type is TokenType.KEYWORD
        assert toks[0].value == "SELECT"
        assert toks[1].type is TokenType.IDENT
        assert toks[1].text == "name"

    def test_case_insensitive_keywords(self) -> None:
        assert tokenize("select")[0].value == "SELECT"
        assert tokenize("SeLeCt")[0].value == "SELECT"

    def test_integer_and_float(self) -> None:
        toks = tokenize("42 3.14 1e3")
        assert toks[0].value == 42
        assert toks[1].value == 3.14
        assert toks[2].value == 1000.0

    def test_string_literal(self) -> None:
        tok = tokenize("'hello'")[0]
        assert tok.type is TokenType.STRING
        assert tok.value == "hello"

    def test_string_with_escaped_quote(self) -> None:
        tok = tokenize("'O''Brien'")[0]
        assert tok.value == "O'Brien"

    def test_quoted_identifier(self) -> None:
        tok = tokenize('"Weird Name"')[0]
        assert tok.type is TokenType.IDENT
        assert tok.text == "Weird Name"

    def test_operators(self) -> None:
        assert _types("<= >= <> != = < >")[:-1] == [
            TokenType.LTE,
            TokenType.GTE,
            TokenType.NEQ,
            TokenType.NEQ,
            TokenType.EQ,
            TokenType.LT,
            TokenType.GT,
        ]

    def test_punctuation(self) -> None:
        assert _types("(),.*")[:-1] == [
            TokenType.LPAREN,
            TokenType.RPAREN,
            TokenType.COMMA,
            TokenType.DOT,
            TokenType.STAR,
        ]

    def test_line_comment_skipped(self) -> None:
        toks = tokenize("SELECT -- a comment\n1")
        assert [t.type for t in toks] == [TokenType.KEYWORD, TokenType.NUMBER, TokenType.EOF]

    def test_ends_with_eof(self) -> None:
        assert tokenize("1")[-1].type is TokenType.EOF

    def test_whitespace_ignored(self) -> None:
        assert len(tokenize("  \n\t 1 \n")) == 2  # NUMBER + EOF


class TestErrors:
    def test_unterminated_string(self) -> None:
        with pytest.raises(LexError, match="unterminated string"):
            tokenize("'oops")

    def test_unterminated_identifier(self) -> None:
        with pytest.raises(LexError, match="unterminated quoted"):
            tokenize('"oops')

    def test_unexpected_character(self) -> None:
        with pytest.raises(LexError, match="unexpected character"):
            tokenize("@")

    def test_token_repr(self) -> None:
        assert "SELECT" in repr(tokenize("SELECT")[0])
