"""The SQL frontend: lexer, parser, and abstract syntax tree.

Turns SQL text into a :class:`~prism.sql.ast.SelectStatement`. The planner
(:mod:`prism.plan`) then binds that tree to the physical operators from
:mod:`prism.exec`, so SQL and the fluent API converge on one execution engine.
"""

from prism.sql.lexer import tokenize
from prism.sql.parser import ParseError, parse

__all__ = ["ParseError", "parse", "tokenize"]
