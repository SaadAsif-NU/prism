"""A fluent, composable query builder over the physical operators.

:class:`Relation` wraps an operator tree and offers chainable transformations,
so a query reads top to bottom:

    >>> from prism import col, load_csv_string, Relation
    >>> people = load_csv_string("name,age\\nAda,36\\nGrace,45\\nAlan,28")
    >>> (
    ...     Relation.from_table(people, "people")
    ...     .filter(col("age") > 30)
    ...     .select("name", col("age"))
    ...     .sort(col("age"), ascending=False)
    ...     .collect()
    ...     .to_rows()
    ... )
    [('Grace', 45), ('Ada', 36)]

The SQL planner (Day 2) builds the very same operator trees, so this is not a
separate engine but a second front door onto the same one.
"""

from __future__ import annotations

from prism.exec.operators import Filter, Limit, Operator, Project, Scan, Sort, SortKey
from prism.expr import Expression, Schema, col
from prism.table import Table


class Relation:
    """A chainable handle on an operator tree."""

    def __init__(self, operator: Operator) -> None:
        self.operator = operator

    @classmethod
    def from_table(cls, table: Table, name: str = "?") -> Relation:
        """Start a query from a materialised table."""
        return cls(Scan(table, name))

    def filter(self, predicate: Expression) -> Relation:
        """Keep rows satisfying ``predicate`` (SQL ``WHERE``)."""
        return Relation(Filter(self.operator, predicate))

    def select(self, *projections: Expression | str) -> Relation:
        """Project a new set of columns (SQL ``SELECT`` list).

        Bare strings are treated as column references, so ``select("name")`` and
        ``select(col("name"))`` are equivalent.
        """
        exprs = [col(p) if isinstance(p, str) else p for p in projections]
        return Relation(Project(self.operator, exprs))

    def sort(
        self,
        *keys: Expression | SortKey,
        ascending: bool = True,
        nulls_first: bool | None = None,
    ) -> Relation:
        """Order rows (SQL ``ORDER BY``).

        Each key may be a :class:`SortKey` or a bare expression; bare
        expressions take the ``ascending`` / ``nulls_first`` defaults passed to
        this call.
        """
        resolved: list[SortKey] = []
        for key in keys:
            if isinstance(key, SortKey):
                resolved.append(key)
            else:
                resolved.append(SortKey(key, ascending=ascending, nulls_first=nulls_first))
        return Relation(Sort(self.operator, resolved))

    def limit(self, limit: int | None, offset: int = 0) -> Relation:
        """Cap the number of rows returned (SQL ``LIMIT`` / ``OFFSET``)."""
        return Relation(Limit(self.operator, limit, offset))

    def collect(self) -> Table:
        """Execute the plan and return the result table."""
        return self.operator.execute()

    def schema(self) -> Schema:
        """The output schema without executing."""
        return self.operator.schema()

    def explain(self) -> str:
        """A textual rendering of the operator tree."""
        return self.operator.explain()

    def __repr__(self) -> str:
        return f"Relation(\n{self.operator.explain(1)}\n)"
