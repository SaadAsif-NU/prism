"""End-to-end tests: SQL text in, result table out."""

from __future__ import annotations

import pytest

from prism import Database, Table
from prism.plan import PlanError
from prism.types import DataType


@pytest.fixture
def db() -> Database:
    database = Database()
    database.register(
        "emp",
        Table.from_pydict(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["Ada", "Grace", "Alan", "Kay", "Margaret"],
                "dept": ["Eng", "Eng", "Research", "Research", "Eng"],
                "salary": [100, 120, 90, 95, 110],
                "age": [36, 45, 41, None, 33],
            },
            types={
                "id": DataType.INTEGER,
                "name": DataType.TEXT,
                "dept": DataType.TEXT,
                "salary": DataType.INTEGER,
                "age": DataType.INTEGER,
            },
        ),
    )
    database.register(
        "dept_info",
        Table.from_pydict(
            {"dept": ["Eng", "Research", "Sales"], "floor": [1, 2, 3]},
            types={"dept": DataType.TEXT, "floor": DataType.INTEGER},
        ),
    )
    return database


def rows(db: Database, sql: str) -> list[tuple]:
    return db.sql(sql).to_rows()


class TestProjection:
    def test_select_columns(self, db: Database) -> None:
        result = db.sql("SELECT name, salary FROM emp")
        assert result.column_names == ["name", "salary"]
        assert result.num_rows == 5

    def test_select_star(self, db: Database) -> None:
        assert db.sql("SELECT * FROM emp").num_columns == 5

    def test_computed_column(self, db: Database) -> None:
        result = db.sql("SELECT salary * 12 AS annual FROM emp")
        assert result.column("annual").to_pylist()[0] == 1200

    def test_alias(self, db: Database) -> None:
        assert db.sql("SELECT name AS who FROM emp").column_names == ["who"]

    def test_select_constant_without_from(self, db: Database) -> None:
        assert rows(db, "SELECT 1 + 1 AS two") == [(2,)]


class TestWhere:
    def test_filter(self, db: Database) -> None:
        assert rows(db, "SELECT name FROM emp WHERE salary >= 110 ORDER BY name") == [
            ("Grace",),
            ("Margaret",),
        ]

    def test_and_or(self, db: Database) -> None:
        result = db.sql("SELECT name FROM emp WHERE dept = 'Eng' AND salary > 105")
        assert set(result.column("name").to_pylist()) == {"Grace", "Margaret"}

    def test_is_null(self, db: Database) -> None:
        assert rows(db, "SELECT name FROM emp WHERE age IS NULL") == [("Kay",)]


class TestOrderLimit:
    def test_order_desc(self, db: Database) -> None:
        assert (
            db.sql("SELECT name FROM emp ORDER BY salary DESC").column("name").to_pylist()[0]
            == "Grace"
        )

    def test_order_by_alias(self, db: Database) -> None:
        result = db.sql("SELECT name, salary * 2 AS s FROM emp ORDER BY s DESC LIMIT 1")
        assert result.column("name").to_pylist() == ["Grace"]

    def test_nulls_last_default(self, db: Database) -> None:
        # age ascending places the NULL (Kay) last
        assert db.sql("SELECT name FROM emp ORDER BY age").column("name").to_pylist()[-1] == "Kay"

    def test_limit_offset(self, db: Database) -> None:
        assert db.sql("SELECT name FROM emp ORDER BY id LIMIT 2 OFFSET 1").column(
            "name"
        ).to_pylist() == ["Grace", "Alan"]


class TestDistinct:
    def test_distinct(self, db: Database) -> None:
        result = db.sql("SELECT DISTINCT dept FROM emp ORDER BY dept")
        assert result.column("dept").to_pylist() == ["Eng", "Research"]


class TestAggregation:
    def test_global_count(self, db: Database) -> None:
        assert rows(db, "SELECT COUNT(*) AS n FROM emp") == [(5,)]

    def test_global_sum_avg(self, db: Database) -> None:
        result = db.sql("SELECT SUM(salary) AS s, AVG(salary) AS a FROM emp")
        assert result.column("s").to_pylist() == [515]
        assert result.column("a").to_pylist() == [103.0]

    def test_group_by(self, db: Database) -> None:
        result = db.sql("SELECT dept, COUNT(*) AS n FROM emp GROUP BY dept ORDER BY dept")
        assert result.to_rows() == [("Eng", 3), ("Research", 2)]

    def test_group_by_names_column(self, db: Database) -> None:
        # The grouped column keeps its own name, not an internal placeholder.
        assert "dept" in db.sql("SELECT dept, COUNT(*) FROM emp GROUP BY dept").column_names

    def test_min_max(self, db: Database) -> None:
        result = db.sql(
            "SELECT dept, MIN(salary) AS lo, MAX(salary) AS hi FROM emp GROUP BY dept ORDER BY dept"
        )
        assert result.to_rows() == [("Eng", 100, 120), ("Research", 90, 95)]

    def test_count_ignores_nulls(self, db: Database) -> None:
        # age has one NULL, so COUNT(age) < COUNT(*)
        assert rows(db, "SELECT COUNT(*) AS all, COUNT(age) AS known FROM emp") == [(5, 4)]

    def test_having(self, db: Database) -> None:
        result = db.sql("SELECT dept FROM emp GROUP BY dept HAVING COUNT(*) > 2")
        assert result.column("dept").to_pylist() == ["Eng"]

    def test_count_distinct(self, db: Database) -> None:
        assert rows(db, "SELECT COUNT(DISTINCT dept) AS d FROM emp") == [(2,)]

    def test_avg_over_null_group(self, db: Database) -> None:
        result = db.sql("SELECT dept, AVG(age) AS a FROM emp GROUP BY dept ORDER BY dept")
        # Research has ages [41, NULL] -> average of the known value only
        assert result.to_rows() == [("Eng", 38.0), ("Research", 41.0)]


class TestScalarFunctions:
    def test_upper_lower(self, db: Database) -> None:
        assert rows(db, "SELECT UPPER(name) FROM emp LIMIT 1") == [("ADA",)]
        assert rows(db, "SELECT LOWER(dept) FROM emp LIMIT 1") == [("eng",)]

    def test_length(self, db: Database) -> None:
        assert rows(db, "SELECT LENGTH(name) AS l FROM emp WHERE name = 'Ada'") == [(3,)]

    def test_coalesce(self, db: Database) -> None:
        result = db.sql("SELECT name, COALESCE(age, 0) AS a FROM emp WHERE name = 'Kay'")
        assert result.column("a").to_pylist() == [0]

    def test_round(self, db: Database) -> None:
        assert rows(db, "SELECT ROUND(AVG(salary), 1) AS a FROM emp") == [(103.0,)]


class TestJoins:
    def test_inner_join(self, db: Database) -> None:
        result = db.sql(
            "SELECT e.name, d.floor FROM emp e JOIN dept_info d ON e.dept = d.dept "
            "ORDER BY e.name LIMIT 1"
        )
        assert result.to_rows() == [("Ada", 1)]

    def test_left_join_keeps_unmatched(self, db: Database) -> None:
        # Sales has no employees; a LEFT join from dept_info keeps it with NULLs.
        result = db.sql(
            "SELECT d.dept, e.name FROM dept_info d LEFT JOIN emp e ON d.dept = e.dept "
            "WHERE d.dept = 'Sales'"
        )
        assert result.to_rows() == [("Sales", None)]

    def test_join_then_group(self, db: Database) -> None:
        result = db.sql(
            "SELECT d.floor, COUNT(*) AS n FROM emp e JOIN dept_info d ON e.dept = d.dept "
            "GROUP BY d.floor ORDER BY d.floor"
        )
        assert result.to_rows() == [(1, 3), (2, 2)]

    def test_cross_join(self, db: Database) -> None:
        result = db.sql("SELECT COUNT(*) AS n FROM emp, dept_info")
        assert result.to_rows() == [(15,)]  # 5 x 3


class TestExplain:
    def test_explain_shows_operators(self, db: Database) -> None:
        text = db.explain(
            "SELECT dept, COUNT(*) FROM emp WHERE salary > 90 GROUP BY dept ORDER BY dept LIMIT 2"
        )
        for op in ("Limit", "Project", "Sort", "HashAggregate", "Filter", "Scan"):
            assert op in text

    def test_explain_join(self, db: Database) -> None:
        text = db.explain("SELECT * FROM emp e JOIN dept_info d ON e.dept = d.dept")
        assert "HashJoin" in text


class TestErrors:
    def test_unknown_table(self, db: Database) -> None:
        with pytest.raises(PlanError, match="unknown table"):
            db.sql("SELECT * FROM nope")

    def test_unknown_column(self, db: Database) -> None:
        with pytest.raises(PlanError, match="unknown column"):
            db.sql("SELECT nope FROM emp")

    def test_non_grouped_column(self, db: Database) -> None:
        with pytest.raises(PlanError, match="GROUP BY"):
            db.sql("SELECT name, COUNT(*) FROM emp GROUP BY dept")

    def test_aggregate_in_where(self, db: Database) -> None:
        with pytest.raises(PlanError, match="aggregate"):
            db.sql("SELECT * FROM emp WHERE COUNT(*) > 1")

    def test_unknown_function(self, db: Database) -> None:
        with pytest.raises(PlanError, match="unknown function"):
            db.sql("SELECT WOBBLE(name) FROM emp")

    def test_ambiguous_column(self, db: Database) -> None:
        with pytest.raises(PlanError, match="ambiguous"):
            db.sql("SELECT dept FROM emp e JOIN dept_info d ON e.dept = d.dept")

    def test_repr(self, db: Database) -> None:
        assert "emp" in repr(db)
