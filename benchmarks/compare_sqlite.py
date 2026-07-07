"""Cross-check prism against SQLite for correctness and timing.

This does two things at once. First it is a *differential test*: every query is
run on both prism and Python's built-in ``sqlite3`` and the results are compared
row for row, so SQLite acts as a reference implementation. Second it reports how
long each engine took, so the cost of prism's pure-Python, vectorized execution
is visible next to a mature C engine.

Run it with the dev environment active::

    python benchmarks/compare_sqlite.py --rows 30000

prism is written entirely in Python and NumPy, so it is not trying to beat
SQLite's C core; the point is that it returns the *same answers* and stays in a
sensible range while doing so.
"""

from __future__ import annotations

import argparse
import sqlite3
import time

import numpy as np

from prism import Database, Table
from prism.types import DataType

REGIONS = ["north", "south", "east", "west"]
PRODUCTS = [f"p{i:02d}" for i in range(10)]

QUERIES = {
    "filter-count": "SELECT COUNT(*) AS n FROM sales WHERE amount > 500",
    "group-by-region": (
        "SELECT region, COUNT(*) AS n, SUM(amount) AS total "
        "FROM sales GROUP BY region ORDER BY region"
    ),
    "group-by-product-avg": (
        "SELECT product, AVG(amount) AS avg_amount, MAX(quantity) AS max_q "
        "FROM sales GROUP BY product ORDER BY product"
    ),
    "filter-project-sort": (
        "SELECT region, amount FROM sales WHERE amount > 900 ORDER BY amount DESC LIMIT 20"
    ),
}


def build_data(rows: int, seed: int = 7) -> dict[str, list]:
    rng = np.random.default_rng(seed)
    return {
        "id": list(range(rows)),
        "region": [REGIONS[i] for i in rng.integers(0, len(REGIONS), rows)],
        "product": [PRODUCTS[i] for i in rng.integers(0, len(PRODUCTS), rows)],
        "amount": [round(float(x), 2) for x in rng.uniform(0, 1000, rows)],
        "quantity": [int(x) for x in rng.integers(1, 50, rows)],
    }


def load_prism(data: dict[str, list]) -> Database:
    db = Database()
    db.register(
        "sales",
        Table.from_pydict(
            data,
            types={
                "id": DataType.INTEGER,
                "region": DataType.TEXT,
                "product": DataType.TEXT,
                "amount": DataType.FLOAT,
                "quantity": DataType.INTEGER,
            },
        ),
    )
    return db


def load_sqlite(data: dict[str, list]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE sales (id INTEGER, region TEXT, product TEXT, amount REAL, quantity INTEGER)"
    )
    rows = zip(*data.values(), strict=True)
    conn.executemany("INSERT INTO sales VALUES (?, ?, ?, ?, ?)", rows)
    conn.commit()
    return conn


def _normalize(rows: list[tuple]) -> list[tuple]:
    out = []
    for row in rows:
        out.append(tuple(round(v, 4) if isinstance(v, float) else v for v in row))
    return out


def _timed(fn) -> tuple[list[tuple], float]:  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=30000)
    parser.add_argument("--repeat", type=int, default=3, help="best-of-N timing")
    args = parser.parse_args()

    data = build_data(args.rows)
    prism_db = load_prism(data)
    sqlite_conn = load_sqlite(data)

    print(f"rows: {args.rows:,}   (best of {args.repeat})\n")
    header = f"{'query':<24}{'prism (ms)':>12}{'sqlite (ms)':>13}{'match':>8}"
    print(header)
    print("-" * len(header))

    all_match = True
    for name, sql in QUERIES.items():
        prism_ms = float("inf")
        sqlite_ms = float("inf")
        prism_rows: list[tuple] = []
        sqlite_rows: list[tuple] = []
        for _ in range(args.repeat):
            rows_p, t_p = _timed(lambda sql=sql: prism_db.sql(sql).to_rows())
            rows_s, t_s = _timed(lambda sql=sql: sqlite_conn.execute(sql).fetchall())
            prism_ms = min(prism_ms, t_p)
            sqlite_ms = min(sqlite_ms, t_s)
            prism_rows, sqlite_rows = rows_p, rows_s

        match = _normalize(prism_rows) == _normalize(sqlite_rows)
        all_match = all_match and match
        flag = "ok" if match else "DIFF"
        print(f"{name:<24}{prism_ms:>12.2f}{sqlite_ms:>13.2f}{flag:>8}")

    print()
    print("all results identical to SQLite" if all_match else "MISMATCH detected")
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
