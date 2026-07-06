# prism

**A columnar SQL query engine, built from scratch.** A typed, nullable column store with Arrow-style validity masks, a hand-written SQL parser, a binding planner, and a vectorised, pull-based execution engine, all in pure Python and NumPy. No database, no ORM, no query library. This is the machinery that lives underneath DuckDB, ClickHouse, and every analytical database, rebuilt small enough to read.

```sql
SELECT department, COUNT(*) AS headcount, ROUND(AVG(salary), 0) AS avg_salary
FROM employees
WHERE salary IS NOT NULL
GROUP BY department
HAVING COUNT(*) > 1
ORDER BY avg_salary DESC
```

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
| Query operators | the database's executor | `exec/`: scan, filter, project, sort, aggregate, join |
| Reading data | a CSV/Parquet reader | `storage/`: a type-inferring CSV loader + catalog |
| SQL | the parser and planner | `sql/` + `plan/`: lexer, parser, binding planner |

## Design that mirrors a real database

**Columnar storage with a validity mask.** A column is a single NumPy array of its native dtype plus a separate boolean mask marking which entries are present. This is exactly how Apache Arrow and pandas' nullable dtypes represent data, and it is what lets an integer column hold NULLs without silently turning into floats or Python objects. Storing values column-by-column keeps a scan over one field cache-friendly and makes whole-column operations trivial to vectorise.

**Vectorised, null-aware expressions.** Every expression node computes an entire output column at once. Arithmetic, comparison, and logical operators all propagate nulls the way SQL requires, including full **three-valued logic**: `TRUE AND NULL` is `NULL`, but `FALSE AND NULL` is `FALSE`, because a known-false operand settles the result regardless of the unknown. Division by zero yields NULL rather than an error, so a single bad row never fails a query.

**One expression tree, two front doors.** The same expression and operator types are produced by the fluent Python API and by the SQL parser. The planner, and later the optimizer, only ever manipulate this shared representation. Nothing downstream knows or cares whether a query arrived as Python or as text.

**A binding planner.** SQL parses into a purely syntactic tree, which the planner then *binds* to execution: it resolves column references against the tables in `FROM` (qualifying names across joins), lowers expression syntax to the vectorised expression tree, extracts equi-join keys from `ON` clauses to choose a hash join over a nested loop, and rewrites `GROUP BY` queries so aggregates and group keys resolve to a hash-aggregation step. This parse-then-bind split is how real databases separate what was written from how it runs.

**A pull-based operator tree.** A query is a tree of operators with a `Scan` at each leaf. Each operator exposes `execute()` to produce its result and `schema()` to report its output types without touching data, so a plan can be type-checked and printed with `EXPLAIN` before it runs.

## Quickstart

```bash
git clone https://github.com/SaadAsif-NU/prism.git
cd prism
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Load a CSV and run SQL against it:

```python
from prism import Database

db = Database()
db.load_csv("data/employees.csv")

result = db.sql("""
    SELECT department, COUNT(*) AS headcount, ROUND(AVG(salary), 0) AS avg_salary
    FROM employees
    GROUP BY department
    ORDER BY avg_salary DESC
""")

for row in result.to_rows():
    print(row)
# ('Engineering', 5, 148800.0)
# ('Research', 5, 143000.0)
```

Joins, filters, and scalar functions all work as you would expect:

```python
db.sql("""
    SELECT e.name, d.building
    FROM employees e
    JOIN departments d ON e.department = d.department
    WHERE e.salary > 145000
    ORDER BY e.name
""")
```

Inspect the physical plan before running it:

```python
print(db.explain("SELECT department, AVG(salary) FROM employees GROUP BY department"))
# Project([department, AVG(salary)])
#   HashAggregate(keys=[__gk0], aggs=[__agg0])
#     Scan(employees, rows=10)
```

The same engine is also reachable through a fluent Python API, since the SQL
planner and the builder produce identical operator trees:

```python
from prism import Relation, col

Relation.from_table(employees, "employees").filter(col("salary") > 145000).select("name").collect()
```

## How it is verified

Correctness is the whole point of building this by hand, so the test suite (250+ tests, CI on Python 3.10 through 3.13, strict mypy, coverage gate) pins down the parts that are easy to get subtly wrong:

- **Null semantics.** The full three-valued truth table for `AND`, `OR`, and `NOT` is tested across all nine combinations of `{TRUE, FALSE, NULL}`, along with null propagation through arithmetic and comparisons, and NULL join keys that never match.
- **Aggregation.** COUNT ignores nulls but COUNT(\*) does not, AVG over an all-null group is NULL, DISTINCT aggregates deduplicate per group, and a global aggregate over an empty input still returns one row.
- **Joins.** Inner and left joins, hash versus nested-loop paths, residual (non-equi) predicates, and null-padding of unmatched left rows.
- **Type inference and ordering.** Integers stay integers, mixed columns widen to FLOAT, division always produces FLOAT, and multi-key sorts are stable with SQL null placement.

## Roadmap

- [x] **Columnar store**: typed, nullable columns with validity masks; immutable buffer-sharing tables
- [x] **Type system**: inference, unification, numeric promotion, SQL-style coercion
- [x] **Vectorised expressions**: arithmetic, comparison, three-valued logic, IS NULL, with a fluent builder
- [x] **Operators**: scan, filter, project, multi-key sort, limit/offset, with EXPLAIN
- [x] **Storage**: type-inferring CSV loader and an in-memory catalog
- [x] **SQL frontend**: a lexer, a precedence-climbing parser, and a binding planner that compiles SQL to these operators
- [x] **Aggregation and joins**: hash GROUP BY with HAVING and DISTINCT, the five core aggregates, hash and nested-loop joins (INNER and LEFT), and scalar functions
- [ ] **Optimizer**: predicate and projection pushdown, constant folding, with before/after EXPLAIN
- [ ] **Interfaces**: an interactive SQL shell and a browser query playground

## Project layout

```
prism/
  types.py            # logical types: inference, unification, promotion
  column.py           # typed nullable column: values buffer + validity mask
  table.py            # immutable ordered set of named columns
  expr.py             # vectorised expression tree + fluent builder
  aggregate.py        # aggregate functions and grouped reductions
  functions.py        # scalar function registry (UPPER, ROUND, COALESCE, ...)
  relation.py         # chainable query builder over the operators
  engine.py           # Database: register tables, run SQL, EXPLAIN
  sql/
    lexer.py          # SQL text -> tokens
    parser.py         # tokens -> AST (recursive descent + precedence climbing)
    ast.py            # the abstract syntax tree
  plan/
    planner.py        # bind the AST to a physical operator tree
  exec/
    operators.py      # scan, filter, project, sort, limit (pull-based)
    aggregate.py      # hash aggregate and distinct
    join.py           # hash join and nested-loop join
  storage/
    csv_loader.py     # CSV -> columnar table with type inference
    catalog.py        # in-memory registry of named tables
tests/                # 250+ tests across every layer
data/                 # sample CSV data
```

## License

MIT
