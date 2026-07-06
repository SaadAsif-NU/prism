"""The catalog: an in-memory registry of named tables the engine can query."""

from __future__ import annotations

from pathlib import Path

from prism.storage.csv_loader import load_csv
from prism.table import Table


class Catalog:
    """Maps table names to their materialised :class:`~prism.table.Table`."""

    def __init__(self) -> None:
        self._tables: dict[str, Table] = {}

    def register(self, name: str, table: Table) -> None:
        """Add or replace a table under ``name``."""
        self._tables[name] = table

    def load_csv(self, path: str | Path, name: str | None = None, **kwargs: object) -> Table:
        """Load a CSV file and register it, defaulting the name to the stem."""
        table = load_csv(path, **kwargs)  # type: ignore[arg-type]
        table_name = name if name is not None else Path(path).stem
        self.register(table_name, table)
        return table

    def get(self, name: str) -> Table:
        """Return the table registered under ``name``."""
        try:
            return self._tables[name]
        except KeyError:
            raise KeyError(f"no table named {name!r} (have {self.names()})") from None

    def drop(self, name: str) -> None:
        """Remove a table from the catalog."""
        self._tables.pop(name, None)

    def names(self) -> list[str]:
        """All registered table names."""
        return list(self._tables)

    def __contains__(self, name: object) -> bool:
        return name in self._tables

    def __repr__(self) -> str:
        return f"Catalog(tables={self.names()})"
