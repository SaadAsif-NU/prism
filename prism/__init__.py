"""prism: a columnar, vectorised SQL query engine built from scratch.

The public surface is intentionally small. Load data into a
:class:`~prism.table.Table`, then query it with the fluent
:class:`~prism.relation.Relation` builder or (from Day 2) with SQL.
"""

from __future__ import annotations

from prism.column import Column
from prism.engine import Database
from prism.exec.operators import SortKey
from prism.expr import Expression, col, lit
from prism.relation import Relation
from prism.storage import Catalog, load_csv, load_csv_string
from prism.table import Table
from prism.types import DataType

__version__ = "0.1.0"

__all__ = [
    "Catalog",
    "Column",
    "Database",
    "DataType",
    "Expression",
    "Relation",
    "SortKey",
    "Table",
    "__version__",
    "col",
    "lit",
    "load_csv",
    "load_csv_string",
]
