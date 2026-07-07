"""Tests for the result formatter and the interactive shell."""

from __future__ import annotations

import io

import pytest

from prism import Database, Table
from prism.cli import Shell, main
from prism.format import render_schema, render_table
from prism.types import DataType


@pytest.fixture
def db() -> Database:
    database = Database()
    database.register(
        "emp",
        Table.from_pydict(
            {"name": ["Ada", "Grace"], "salary": [100, None]},
            types={"name": DataType.TEXT, "salary": DataType.INTEGER},
        ),
    )
    return database


class TestRenderTable:
    def test_headers_and_types(self) -> None:
        t = Table.from_pydict({"a": [1], "b": ["x"]})
        text = render_table(t)
        assert "a" in text and "b" in text
        assert "INTEGER" in text and "TEXT" in text

    def test_null_rendered(self) -> None:
        t = Table.from_pydict({"a": [1, None]}, types={"a": DataType.INTEGER})
        assert "NULL" in render_table(t)

    def test_row_count_summary(self) -> None:
        t = Table.from_pydict({"a": [1, 2, 3]})
        assert "3 rows in set" in render_table(t)

    def test_single_row_singular(self) -> None:
        t = Table.from_pydict({"a": [1]})
        assert "1 row in set" in render_table(t)

    def test_max_rows_elision(self) -> None:
        t = Table.from_pydict({"a": list(range(100))})
        text = render_table(t, max_rows=10)
        assert "100 rows (10 shown)" in text

    def test_boolean_rendering(self) -> None:
        t = Table.from_pydict({"flag": [True, False]}, types={"flag": DataType.BOOLEAN})
        text = render_table(t)
        assert "true" in text and "false" in text

    def test_float_formatting(self) -> None:
        t = Table.from_pydict({"x": [3.14159]}, types={"x": DataType.FLOAT})
        assert "3.14159" in render_table(t)

    def test_long_text_truncated(self) -> None:
        t = Table.from_pydict({"s": ["x" * 100]})
        assert "…" in render_table(t)

    def test_no_columns(self) -> None:
        assert render_table(Table([])) == "(no columns)"


class TestRenderSchema:
    def test_lists_columns(self, db: Database) -> None:
        text = render_schema("emp", db.catalog.get("emp"))
        assert "emp" in text and "salary" in text and "INTEGER" in text


class TestShell:
    def _shell(self, db: Database) -> tuple[Shell, io.StringIO]:
        out = io.StringIO()
        return Shell(db, timing=False, out=out), out

    def test_select_renders(self, db: Database) -> None:
        shell, out = self._shell(db)
        shell.run_statement("SELECT name FROM emp;")
        assert "Ada" in out.getvalue()

    def test_tables_command(self, db: Database) -> None:
        shell, out = self._shell(db)
        shell.run_statement(".tables")
        assert "emp" in out.getvalue()

    def test_schema_command(self, db: Database) -> None:
        shell, out = self._shell(db)
        shell.run_statement(".schema emp")
        assert "salary" in out.getvalue()

    def test_schema_unknown_table(self, db: Database) -> None:
        shell, out = self._shell(db)
        shell.run_statement(".schema nope")
        assert "unknown table" in out.getvalue()

    def test_explain_command(self, db: Database) -> None:
        shell, out = self._shell(db)
        shell.run_statement("EXPLAIN SELECT name FROM emp;")
        assert "Scan" in out.getvalue()

    def test_timing_toggle(self, db: Database) -> None:
        shell, out = self._shell(db)
        shell.run_statement(".timing off")
        assert "timing off" in out.getvalue()

    def test_unknown_command(self, db: Database) -> None:
        shell, out = self._shell(db)
        shell.run_statement(".bogus")
        assert "unknown command" in out.getvalue()

    def test_empty_statement_ignored(self, db: Database) -> None:
        shell, out = self._shell(db)
        shell.run_statement("   ")
        assert out.getvalue() == ""

    def test_load_command(self, db: Database, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "extra.csv"
        path.write_text("x,y\n1,2\n3,4")
        shell, out = self._shell(db)
        shell.run_statement(f".load {path} extra")
        assert "loaded 'extra'" in out.getvalue()
        assert "extra" in db.catalog


class TestMain:
    def test_command_mode(self, capsys) -> None:  # type: ignore[no-untyped-def]
        rc = main(["-c", "SELECT 1 AS one", "--no-timing"])
        assert rc == 0
        assert "one" in capsys.readouterr().out

    def test_command_mode_error(self, capsys) -> None:  # type: ignore[no-untyped-def]
        rc = main(["-c", "SELECT * FROM missing"])
        assert rc == 1
        assert "error" in capsys.readouterr().err

    def test_loads_csv_positional(self, tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "nums.csv"
        path.write_text("a\n1\n2")
        rc = main([str(path), "-c", "SELECT COUNT(*) AS c FROM nums", "--no-timing"])
        assert rc == 0
        assert "c" in capsys.readouterr().out
