"""The database facade: register tables, run SQL, inspect plans.

:class:`Database` is the one object most users need. It owns a catalog, accepts
data from CSV or in-memory tables, and turns a SQL string into a result table by
running it through the parser, planner, and execution engine.
"""

from __future__ import annotations

from pathlib import Path

from prism.plan import plan
from prism.sql import parse
from prism.storage.catalog import Catalog
from prism.table import Table


class Database:
    """An in-memory SQL database over columnar tables."""

    def __init__(self, catalog: Catalog | None = None) -> None:
        self.catalog = catalog if catalog is not None else Catalog()

    def register(self, name: str, table: Table) -> None:
        """Make ``table`` queryable under ``name``."""
        self.catalog.register(name, table)

    def load_csv(self, path: str | Path, name: str | None = None, **kwargs: object) -> Table:
        """Load a CSV file, register it, and return the table."""
        return self.catalog.load_csv(path, name=name, **kwargs)

    def sql(self, query: str) -> Table:
        """Run a SQL query and return the result as a table."""
        return plan(parse(query), self.catalog).execute()

    def explain(self, query: str) -> str:
        """Return the physical plan for ``query`` without running it."""
        return plan(parse(query), self.catalog).explain()

    def __repr__(self) -> str:
        return f"Database(tables={self.catalog.names()})"
