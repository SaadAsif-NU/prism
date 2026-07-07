"""The database facade: register tables, run SQL, inspect plans.

:class:`Database` is the one object most users need. It owns a catalog, accepts
data from CSV or in-memory tables, and turns a SQL string into a result table by
running it through the parser, planner, and execution engine.
"""

from __future__ import annotations

from pathlib import Path

from prism.exec.operators import Operator
from prism.plan import optimize, plan
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

    def plan(self, query: str, optimized: bool = True) -> Operator:
        """Parse and plan ``query`` into an operator tree.

        With ``optimized`` set (the default), the rule-based optimizer is
        applied; pass ``False`` to get the raw plan straight from the binder.
        """
        tree = plan(parse(query), self.catalog)
        return optimize(tree) if optimized else tree

    def sql(self, query: str) -> Table:
        """Run a SQL query and return the result as a table."""
        return self.plan(query, optimized=True).execute()

    def explain(self, query: str, optimized: bool = True) -> str:
        """Return the physical plan for ``query`` without running it."""
        return self.plan(query, optimized=optimized).explain()

    def explain_diff(self, query: str) -> str:
        """Show the plan before and after optimization, side by side in text."""
        original = self.plan(query, optimized=False).explain()
        improved = self.plan(query, optimized=True).explain()
        return f"-- original plan --\n{original}\n\n-- optimized plan --\n{improved}"

    def __repr__(self) -> str:
        return f"Database(tables={self.catalog.names()})"
