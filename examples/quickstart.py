"""A tour of prism: load data, run SQL, use the fluent API, see the optimizer.

Run it from the repository root:

    python examples/quickstart.py
"""

from __future__ import annotations

from pathlib import Path

from prism import Database, Relation, col

DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    db = Database()
    db.load_csv(DATA / "employees.csv")
    db.load_csv(DATA / "departments.csv")

    print("== group by with an aggregate ==")
    result = db.sql("""
        SELECT department,
               COUNT(*)            AS headcount,
               ROUND(AVG(salary), 0) AS avg_salary
        FROM employees
        WHERE salary IS NOT NULL
        GROUP BY department
        ORDER BY avg_salary DESC
    """)
    for row in result.to_rows():
        print(row)

    print("\n== a join across two tables ==")
    result = db.sql("""
        SELECT e.name, e.salary, d.location
        FROM employees e
        JOIN departments d ON e.department = d.name
        WHERE e.remote = true
        ORDER BY e.salary DESC NULLS LAST
    """)
    for row in result.to_rows():
        print(row)

    print("\n== the same query through the fluent API ==")
    employees = db.catalog.get("employees")
    rows = (
        Relation.from_table(employees, "employees")
        .filter(col("salary") > 145000)
        .select("name", "salary")
        .sort(col("salary"), ascending=False)
        .collect()
        .to_rows()
    )
    for row in rows:
        print(row)

    print("\n== the optimizer, before and after ==")
    print(db.explain_diff("SELECT name FROM employees WHERE salary > 140000 AND 1 = 1"))


if __name__ == "__main__":
    main()
