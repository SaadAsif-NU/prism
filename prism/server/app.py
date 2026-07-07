"""FastAPI application backing the SQL playground.

Endpoints are deliberately small and stateless: each request names a SQL string,
and the server parses, plans, optimizes, and runs it against a shared in-memory
:class:`~prism.engine.Database`. Results are serialised to plain JSON (NULLs as
``null``, non-finite floats coerced to ``null``) so the browser can render them
directly.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from prism import __version__
from prism.engine import Database
from prism.plan import PlanError
from prism.sql.lexer import LexError
from prism.sql.parser import ParseError
from prism.table import Table

_STATIC_DIR = Path(__file__).parent / "static"
_QUERY_ERRORS = (ParseError, LexError, PlanError, KeyError, TypeError, ValueError)

SAMPLE_QUERIES = [
    {
        "label": "Top earners",
        "sql": (
            "SELECT name, department, salary\n"
            "FROM employees\n"
            "ORDER BY salary DESC NULLS LAST\n"
            "LIMIT 5;"
        ),
    },
    {
        "label": "Average salary by department",
        "sql": (
            "SELECT department,\n"
            "       COUNT(*) AS headcount,\n"
            "       AVG(salary) AS avg_salary\n"
            "FROM employees\n"
            "GROUP BY department\n"
            "ORDER BY avg_salary DESC;"
        ),
    },
    {
        "label": "Join with departments",
        "sql": (
            "SELECT e.name, e.salary, d.location\n"
            "FROM employees e\n"
            "JOIN departments d ON e.department = d.name\n"
            "WHERE e.remote = true\n"
            "ORDER BY e.salary DESC;"
        ),
    },
    {
        "label": "Filter and compute",
        "sql": (
            "SELECT name,\n"
            "       salary,\n"
            "       ROUND(salary / 12, 2) AS monthly\n"
            "FROM employees\n"
            "WHERE salary > 140000\n"
            "ORDER BY monthly DESC;"
        ),
    },
]


def _clean(value: Any) -> Any:
    """Coerce a Python value into something JSON can represent."""
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def serialize_table(table: Table) -> dict[str, Any]:
    """Turn a result table into a JSON-friendly payload for the UI."""
    return {
        "columns": table.column_names,
        "types": [dtype.value for _, dtype in table.schema],
        "rows": [[_clean(v) for v in row] for row in table.to_rows()],
        "row_count": table.num_rows,
    }


def _describe_tables(db: Database) -> list[dict[str, Any]]:
    tables = []
    for name in sorted(db.catalog.names()):
        table = db.catalog.get(name)
        tables.append(
            {
                "name": name,
                "rows": table.num_rows,
                "columns": [{"name": col, "type": dtype.value} for col, dtype in table.schema],
            }
        )
    return tables


def create_app(db: Database) -> Any:
    """Build the FastAPI app serving ``db``.

    Imported lazily so the core package never hard-depends on FastAPI.
    """
    from fastapi import Body, FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="prism playground", version=__version__)

    def _sql_of(body: dict) -> str:
        sql = body.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("request body must include a non-empty 'sql' string")
        return sql

    @app.get("/api/tables")
    def list_tables() -> dict[str, Any]:
        return {"tables": _describe_tables(db)}

    @app.get("/api/samples")
    def samples() -> dict[str, Any]:
        return {"samples": SAMPLE_QUERIES}

    @app.post("/api/query")
    def run_query(body: dict = Body(...)) -> Any:  # noqa: B008
        start = time.perf_counter()
        try:
            result = db.sql(_sql_of(body))
        except _QUERY_ERRORS as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        elapsed_ms = (time.perf_counter() - start) * 1000
        payload = serialize_table(result)
        payload.update(ok=True, elapsed_ms=round(elapsed_ms, 3))
        return payload

    @app.post("/api/explain")
    def explain(body: dict = Body(...)) -> Any:  # noqa: B008
        try:
            sql = _sql_of(body)
            optimized = db.explain(sql, optimized=True)
            original = db.explain(sql, optimized=False)
        except _QUERY_ERRORS as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return {"ok": True, "optimized": optimized, "original": original}

    if _STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

    return app


def default_database(csv_paths: list[str] | None = None) -> Database:
    """A database seeded with the bundled sample data (or the given CSVs)."""
    db = Database()
    if csv_paths:
        for path in csv_paths:
            db.load_csv(path)
        return db
    data_dir = Path(__file__).resolve().parents[2] / "data"
    for name in ("employees", "departments"):
        csv = data_dir / f"{name}.csv"
        if csv.exists():
            db.load_csv(csv, name=name)
    return db


def serve(
    csv_paths: list[str] | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Launch the playground web server (requires the ``server`` extra)."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise SystemExit(
            "the web playground needs extra packages; install them with:\n"
            "    pip install 'prism-sql[server]'"
        ) from exc

    app = create_app(default_database(csv_paths))
    print(f"prism playground on http://{host}:{port}  (Ctrl+C to stop)")
    uvicorn.run(app, host=host, port=port, log_level="warning")
