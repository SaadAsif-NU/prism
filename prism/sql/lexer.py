"""The SQL lexer: turns a query string into a stream of tokens.

A small hand-written scanner. It recognises numbers (integer and float),
single-quoted string literals (with ``''`` as an escaped quote), quoted and
unquoted identifiers, the reserved keywords, and the operator/punctuation set.
``--`` line comments are skipped.
"""

from __future__ import annotations

from prism.sql.tokens import KEYWORDS, Token, TokenType

# Multi-character operators must be tried before their single-character
# prefixes, so this list is ordered longest-first.
_TWO_CHAR = {
    "<=": TokenType.LTE,
    ">=": TokenType.GTE,
    "<>": TokenType.NEQ,
    "!=": TokenType.NEQ,
}
_ONE_CHAR = {
    "*": TokenType.STAR,
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "/": TokenType.SLASH,
    "%": TokenType.PERCENT,
    "=": TokenType.EQ,
    "<": TokenType.LT,
    ">": TokenType.GT,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    ",": TokenType.COMMA,
    ".": TokenType.DOT,
    ";": TokenType.SEMICOLON,
}


class LexError(ValueError):
    """Raised when the input contains a character the lexer cannot handle."""


def tokenize(sql: str) -> list[Token]:
    """Scan ``sql`` into a list of tokens, terminated by an EOF token."""
    tokens: list[Token] = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]

        if ch in " \t\r\n":
            i += 1
            continue

        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            while i < n and sql[i] != "\n":
                i += 1
            continue

        if ch.isdigit() or (ch == "." and i + 1 < n and sql[i + 1].isdigit()):
            token, i = _scan_number(sql, i)
            tokens.append(token)
            continue

        if ch == "'":
            token, i = _scan_string(sql, i)
            tokens.append(token)
            continue

        if ch == '"':
            token, i = _scan_quoted_ident(sql, i)
            tokens.append(token)
            continue

        if ch.isalpha() or ch == "_":
            token, i = _scan_word(sql, i)
            tokens.append(token)
            continue

        two = sql[i : i + 2]
        if two in _TWO_CHAR:
            tokens.append(Token(_TWO_CHAR[two], two, i))
            i += 2
            continue

        if ch in _ONE_CHAR:
            tokens.append(Token(_ONE_CHAR[ch], ch, i))
            i += 1
            continue

        raise LexError(f"unexpected character {ch!r} at position {i}")

    tokens.append(Token(TokenType.EOF, "", n))
    return tokens


def _scan_number(sql: str, start: int) -> tuple[Token, int]:
    i = start
    n = len(sql)
    seen_dot = False
    seen_exp = False
    while i < n:
        ch = sql[i]
        if ch.isdigit():
            i += 1
        elif ch == "." and not seen_dot and not seen_exp:
            seen_dot = True
            i += 1
        elif ch in "eE" and not seen_exp:
            seen_exp = True
            i += 1
            if i < n and sql[i] in "+-":
                i += 1
        else:
            break
    text = sql[start:i]
    is_float = seen_dot or seen_exp
    value: object = float(text) if is_float else int(text)
    return Token(TokenType.NUMBER, text, start, value), i


def _scan_string(sql: str, start: int) -> tuple[Token, int]:
    i = start + 1
    n = len(sql)
    chars: list[str] = []
    while i < n:
        ch = sql[i]
        if ch == "'":
            if i + 1 < n and sql[i + 1] == "'":  # '' is an escaped quote
                chars.append("'")
                i += 2
                continue
            return Token(TokenType.STRING, sql[start : i + 1], start, "".join(chars)), i + 1
        chars.append(ch)
        i += 1
    raise LexError(f"unterminated string literal starting at position {start}")


def _scan_quoted_ident(sql: str, start: int) -> tuple[Token, int]:
    i = start + 1
    n = len(sql)
    while i < n:
        if sql[i] == '"':
            return Token(TokenType.IDENT, sql[start + 1 : i], start), i + 1
        i += 1
    raise LexError(f"unterminated quoted identifier starting at position {start}")


def _scan_word(sql: str, start: int) -> tuple[Token, int]:
    i = start
    n = len(sql)
    while i < n and (sql[i].isalnum() or sql[i] == "_"):
        i += 1
    text = sql[start:i]
    upper = text.upper()
    if upper in KEYWORDS:
        return Token(TokenType.KEYWORD, text, start, upper), i
    return Token(TokenType.IDENT, text, start), i
