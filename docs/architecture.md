# Architecture

prism is a small analytical database. This document follows a single query from
text to result and explains why each stage exists.

## The lifecycle of a query

```
SQL text
   │  sql/lexer.py        tokenize
   ▼
tokens
   │  sql/parser.py       precedence-climbing recursive descent
   ▼
AST (sql/ast.py)          purely syntactic; column refs keep qualifiers
   │  plan/planner.py     bind names, lower expressions, choose join strategy
   ▼
operator tree (exec/)     logical == physical: one tree
   │  plan/optimizer.py   fold, push predicates down, prune columns
   ▼
optimized operator tree
   │  Operator.execute()  pull-based, vectorized
   ▼
result Table
```

The two ideas that shape everything are **columnar storage** and **vectorized
execution**. Data lives column by column, and operators transform whole columns
at a time rather than looping over rows.

## Storage: columns and tables

A [`Column`](../prism/column.py) is a single NumPy array of a fixed dtype
(`int64`, `float64`, `bool`, or object for text) paired with a boolean
**validity mask** marking which entries are present. This is the Apache Arrow
representation. Keeping nullability in a side mask is what lets an integer column
hold NULLs without decaying to float or object, and it makes null-aware kernels a
matter of combining masks.

A [`Table`](../prism/table.py) is an ordered set of equal-length named columns.
Tables are immutable: `filter`, `take`, and `slice` return new tables that share
the underlying column buffers, so plans compose without copying data.

## Expressions: one tree, two front doors

An [`Expression`](../prism/expr.py) computes one output column from an input
table. The same tree is built two ways: by the fluent builder (`col("age") >
30`) and by the SQL parser. Because both converge on this representation, the
planner, optimizer, and executor never deal with SQL syntax.

Evaluation is vectorized and null-aware. Logical operators implement SQL's
**three-valued logic**: `TRUE AND NULL` is `NULL`, but `FALSE AND NULL` is
`FALSE`, because a known-false operand settles the result. The mask arithmetic
for this lives in `_eval_logical`.

## SQL frontend: lex, parse, bind

The [lexer](../prism/sql/lexer.py) is a hand-written scanner. The
[parser](../prism/sql/parser.py) is recursive descent with **precedence
climbing** for expressions, so `a OR b AND c = d + 1` associates correctly
without a grammar generator. It produces a purely syntactic
[AST](../prism/sql/ast.py).

The [planner](../prism/plan/planner.py) is the *binder*: it turns syntax into
execution. It resolves column references against the tables in `FROM` (qualifying
names like `e.salary` across joins and rejecting ambiguous bare names), lowers
AST expressions to the vectorized expression tree, extracts equi-join keys from
`ON` clauses to pick a hash join over a nested loop, and rewrites `GROUP BY`
queries so aggregates and group keys resolve against a hash-aggregation step.
This parse-then-bind split is how real databases separate what was written from
how it runs.

## Execution: a pull-based operator tree

A query is a tree of [operators](../prism/exec/) with a `Scan` at each leaf.
Every operator exposes `execute()` to produce a table and `schema()` to report
its output types without touching data. The core operators are scan, filter,
project, sort (stable, multi-key, SQL null placement), limit,
[hash aggregate](../prism/exec/aggregate.py), distinct, and
[joins](../prism/exec/join.py) (hash for equi-joins, nested loop otherwise;
INNER and LEFT).

Because `schema()` is data-free, a plan can be type-checked and printed with
`EXPLAIN` before it runs.

## The optimizer

Since the logical and physical plan are the same tree, the
[optimizer](../prism/plan/optimizer.py) rewrites it directly, applying three
rules to a fixpoint:

1. **Constant folding.** A subexpression with no column inputs is evaluated once
   and replaced by a literal; boolean identities like `x AND TRUE` are
   simplified. This is also what removes a `WHERE 1 = 1`.
2. **Predicate pushdown.** Filters move toward the scans: through projections
   (substituting aliases), through sorts, and split across joins so each
   conjunct runs on the side that owns its columns. A LEFT join keeps
   right-side predicates above it, because pushing them would change the
   answer.
3. **Column pruning.** Walking the tree with the set of columns each subtree
   needs, a narrow projection is inserted at each scan, so a columnar read never
   materializes a column the query ignores.

Every rule returns a new tree; nothing is mutated. `explain_diff` prints the
plan before and after so the rewrites are visible.

## Why it is trustworthy

Every rewrite is checked two ways: structurally (the plan changed as intended)
and semantically (the optimized plan returns exactly the rows the unoptimized
plan does). On top of that, `benchmarks/compare_sqlite.py` runs queries on both
prism and SQLite and compares results row for row, using a mature engine as a
reference oracle.
