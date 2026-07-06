"""prism: a columnar, vectorised SQL query engine built from scratch.

The public surface is intentionally small. Load data into a
:class:`~prism.table.Table`, then query it with the fluent
:class:`~prism.relation.Relation` builder or (from Day 2) with SQL.
"""

from __future__ import annotations

from prism.column import Column
from prism.expr import Expression, col, lit
from prism.table import Table
from prism.types import DataType

__version__ = "0.1.0"

__all__ = [
    "Column",
    "DataType",
    "Expression",
    "Table",
    "__version__",
    "col",
    "lit",
]
