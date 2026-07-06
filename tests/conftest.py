"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest

from prism import Table
from prism.types import DataType


@pytest.fixture
def people() -> Table:
    """A small table with an integer, text, and nullable numeric column."""
    return Table.from_pydict(
        {
            "name": ["Ada", "Grace", "Alan", "Katherine", "Margaret"],
            "age": [36, 45, 41, None, 33],
            "city": ["London", "New York", "London", "Hampton", None],
            "salary": [145000.0, 152000.0, 138000.0, 141000.0, 149000.0],
        },
        types={
            "name": DataType.TEXT,
            "age": DataType.INTEGER,
            "city": DataType.TEXT,
            "salary": DataType.FLOAT,
        },
    )
