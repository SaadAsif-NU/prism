# prism

**A columnar SQL query engine, built from scratch.** A typed, nullable column store with Arrow-style validity masks, a vectorised expression engine, and a pull-based operator tree, all in pure Python and NumPy. No database, no ORM, no query library. This is the machinery that lives underneath DuckDB, ClickHouse, and every analytical database, rebuilt small enough to read.

[![CI](https://github.com/SaadAsif-NU/prism/actions/workflows/ci.yml/badge.svg)](https://github.com/SaadAsif-NU/prism/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/SaadAsif-NU/prism)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## The idea

Analytical databases are fast because of two decisions that show up everywhere in their design: they store data **by column instead of by row**, and they execute queries **a column at a time instead of a row at a time**. prism is those two decisions taken seriously and built from the ground up.

| The job | What you'd normally import | What prism implements instead |
|---|---|---|
| Typed nullable storage | pandas / Arrow | `column.py`: a values buffer plus a validity mask |
| Tables and schemas | a dataframe library | `table.py`: immutable, buffer-sharing tables |
| Expressions | the database's evaluator | `expr.py`: a vectorised expression tree |
| Query operators | the database's executor | `exec/operators.py`: scan, filter, project, sort, limit |
| Reading data | a CSV/Parquet reader | `storage/`: a type-inferring CSV loader + catalog |
| SQL | the parser and planner | Day 2: lexer, parser, logical planner |

## Design that mirrors a real database

**Columnar storage with a validity mask.** A column is a single NumPy array of its native dtype plus a separate boolean mask marking which entries are present. This is exactly how Apache Arrow and pandas' nullable dtypes represent data, and it is what lets an integer column hold NULLs without silently turning into floats or Python objects. Storing values column-by-column keeps a scan over one field cache-friendly and makes whole-column operations trivial to vectorise.

**Vectorised, null-aware expressions.** Every expression node computes an entire output column at once. Arithmetic, comparison, and logical operators all propagate nulls the way SQL requires, including full **three-valued logic**: `TRUE AND NULL` is `NULL`, but `FALSE AND NULL` is `FALSE`, because a known-false operand settles the result regardless of the unknown. Division by zero yields NULL rather than an error, so a single bad row never fails a query.

**One expression tree, two front doors.** The same expression and operator types are produced by the fluent Python API today and by the SQL parser on Day 2. The planner, and later the optimizer, only ever manipulate this shared representation. Nothing downstream knows or cares whether a query arrived as Python or as text.

**A pull-based operator tree.** A query is a tree of operators with a `Scan` at each leaf. Each operator exposes `execute()` to produce its result and `schema()` to report its output types without touching data, so a plan can be type-checked and printed with `EXPLAIN` before it runs.

## Quickstart

```bash
git clone https://github.com/SaadAsif-NU/prism.git
cd prism
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Load a CSV and query it with the fluent API:

```python
from prism import Catalog, Relation, col

catalog = Catalog()
employees = catalog.load_csv("data/employees.csv")

result = (
    Relation.from_table(employees, "employees")
    .filter((col("department") == "Engineering") & (col("salary") > 145000))
    .select("name", "salary", (col("salary") / 12).alias("monthly"))
    .sort(col("salary"), ascending=False)
    .collect()
)

for row in result.to_rows():
    print(row)
# ('Grace Hopper', 152000, 12666.67)
# ('Barbara Liskov', 151000, 12583.33)
# ('Margaret Hamilton', 149000, 12416.67)
# ('Dennis Ritchie', 147000, 12250.0)
```

Inspect the plan before running it:

```python
plan = Relation.from_table(employees, "employees").filter(col("salary") > 145000).select("name")
print(plan.explain())
# Project([name])
#   Filter((salary > 145000))
#     Scan(employees, rows=10)
```

## How it is verified

Correctness is the whole point of building this by hand, so the test suite (150+ tests, CI on Python 3.10 through 3.13, strict mypy, coverage gate) pins down the parts that are easy to get subtly wrong:

- **Null semantics.** The full three-valued truth table for `AND`, `OR`, and `NOT` is tested across all nine combinations of `{TRUE, FALSE, NULL}`, along with null propagation through arithmetic and comparisons.
- **Type inference.** Integers stay integers, mixed integer/float columns widen to FLOAT, all-empty columns resolve to TEXT, and division always produces FLOAT.
- **Ordering.** Multi-key sorts are stable, respect per-key ASC/DESC, and place NULLs using SQL's convention (as if larger than any value).
- **Storage.** Take, filter, and slice preserve values and their validity masks together.

## Roadmap

- [x] **Columnar store**: typed, nullable columns with validity masks; immutable buffer-sharing tables
- [x] **Type system**: inference, unification, numeric promotion, SQL-style coercion
- [x] **Vectorised expressions**: arithmetic, comparison, three-valued logic, IS NULL, with a fluent builder
- [x] **Operators**: scan, filter, project, multi-key sort, limit/offset, with EXPLAIN
- [x] **Storage**: type-inferring CSV loader and an in-memory catalog
- [ ] **SQL frontend**: lexer, parser, and a logical planner that compiles SQL to these operators
- [ ] **Aggregation and joins**: hash GROUP BY with HAVING, hash joins
- [ ] **Optimizer**: predicate and projection pushdown, constant folding, with before/after EXPLAIN
- [ ] **Interfaces**: an interactive SQL shell and a browser query playground

## Project layout

```
prism/
  types.py            # logical types: inference, unification, promotion
  column.py           # typed nullable column: values buffer + validity mask
  table.py            # immutable ordered set of named columns
  expr.py             # vectorised expression tree + fluent builder
  relation.py         # chainable query builder over the operators
  exec/
    operators.py      # scan, filter, project, sort, limit (pull-based)
  storage/
    csv_loader.py     # CSV -> columnar table with type inference
    catalog.py        # in-memory registry of named tables
tests/                # null-semantics, inference, sorting, storage
data/                 # sample CSV data
```

## License

MIT
