"""Token definitions for the SQL lexer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TokenType(Enum):
    """The lexical category of a token."""

    NUMBER = "NUMBER"
    STRING = "STRING"
    IDENT = "IDENT"
    KEYWORD = "KEYWORD"
    # operators and punctuation
    STAR = "*"
    PLUS = "+"
    MINUS = "-"
    SLASH = "/"
    PERCENT = "%"
    EQ = "="
    NEQ = "<>"
    LT = "<"
    LTE = "<="
    GT = ">"
    GTE = ">="
    LPAREN = "("
    RPAREN = ")"
    COMMA = ","
    DOT = "."
    SEMICOLON = ";"
    EOF = "EOF"


#: Reserved words. Anything matching one of these (case-insensitively) lexes as
#: a KEYWORD rather than an identifier. Aggregate and scalar function names are
#: deliberately *not* reserved; they are ordinary identifiers recognised as
#: calls by the parser when followed by "(".
KEYWORDS = frozenset(
    {
        "SELECT",
        "FROM",
        "WHERE",
        "GROUP",
        "BY",
        "HAVING",
        "ORDER",
        "ASC",
        "DESC",
        "NULLS",
        "FIRST",
        "LAST",
        "LIMIT",
        "OFFSET",
        "AS",
        "AND",
        "OR",
        "NOT",
        "IS",
        "NULL",
        "TRUE",
        "FALSE",
        "DISTINCT",
        "JOIN",
        "INNER",
        "LEFT",
        "RIGHT",
        "OUTER",
        "ON",
    }
)


@dataclass(frozen=True)
class Token:
    """A single lexical token with its source position."""

    type: TokenType
    text: str
    pos: int
    # For NUMBER/STRING this holds the parsed Python value; for KEYWORD the
    # upper-cased word; otherwise the raw text.
    value: object = None

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.text!r})"
