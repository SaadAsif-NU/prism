# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Columnar core.** Typed, nullable columns backed by a NumPy values buffer and
  an Arrow-style validity mask; immutable, buffer-sharing tables; a logical type
  system with inference, unification, and numeric promotion.
- **Vectorized expressions.** An expression tree evaluated a column at a time,
  with SQL three-valued logic for `AND`/`OR`/`NOT`, null propagation through
  arithmetic and comparisons, `IS NULL`, and a fluent Python builder.
- **Physical operators.** Scan, filter, project, stable multi-key sort with SQL
  null placement, limit/offset, hash `GROUP BY` aggregation (COUNT, SUM, AVG,
  MIN, MAX, with `DISTINCT` and `HAVING`), `DISTINCT`, and joins (hash for
  equi-joins, nested loop otherwise; INNER and LEFT).
- **SQL frontend.** A hand-written lexer, a precedence-climbing recursive-descent
  parser, and a binding planner that resolves qualified column references,
  lowers syntax to the expression tree, and chooses a join strategy.
- **Scalar functions.** `UPPER`, `LOWER`, `LENGTH`, `ABS`, `ROUND`, `COALESCE`.
- **Storage.** A type-inferring CSV loader and an in-memory catalog.
- **Rule-based optimizer.** Constant folding and boolean simplification,
  predicate pushdown (through projections and sorts, split across joins), and
  column pruning at scans, applied to a fixpoint. `EXPLAIN` and `explain_diff`
  show plans before and after.
- **Interactive SQL shell.** The `prism` command: a REPL with box-drawing result
  tables, query timing, `.tables` / `.schema` / `.load` introspection, and
  `EXPLAIN`.
- **Browser query playground.** An optional FastAPI app (`prism --serve`, behind
  the `server` extra) with a SQL editor, typed result grid, and a plan visualizer
  that shows the optimized tree next to the original.
- **SQLite differential benchmark.** `benchmarks/compare_sqlite.py` cross-checks
  results against SQLite and reports timing.

## [0.1.0]

- Initial release.
