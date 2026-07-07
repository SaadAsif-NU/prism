"""Tests for the web playground's HTTP API.

These use FastAPI's TestClient, which is only available with the ``server``
extra installed; the whole module is skipped otherwise so the core test run
never depends on it.
"""

from __future__ import annotations

import pytest

from prism import Table
from prism.engine import Database

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from prism.server.app import create_app, default_database, serialize_table  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    db = Database()
    db.register(
        "emp",
        Table.from_pydict({"name": ["Ada", "Grace"], "salary": [100, 200], "dept": ["Eng", "Eng"]}),
    )
    return TestClient(create_app(db))


class TestSerialize:
    def test_nulls_and_types(self) -> None:
        from prism.types import DataType

        table = Table.from_pydict({"a": [1, None]}, types={"a": DataType.INTEGER})
        payload = serialize_table(table)
        assert payload["columns"] == ["a"]
        assert payload["types"] == ["INTEGER"]
        assert payload["rows"] == [[1], [None]]
        assert payload["row_count"] == 2


class TestApi:
    def test_list_tables(self, client: TestClient) -> None:
        data = client.get("/api/tables").json()
        assert data["tables"][0]["name"] == "emp"
        assert {"name": "salary", "type": "INTEGER"} in data["tables"][0]["columns"]

    def test_samples(self, client: TestClient) -> None:
        data = client.get("/api/samples").json()
        assert len(data["samples"]) >= 1
        assert "sql" in data["samples"][0]

    def test_query_ok(self, client: TestClient) -> None:
        data = client.post("/api/query", json={"sql": "SELECT name FROM emp"}).json()
        assert data["ok"] is True
        assert data["columns"] == ["name"]
        assert data["rows"] == [["Ada"], ["Grace"]]
        assert "elapsed_ms" in data

    def test_query_aggregate(self, client: TestClient) -> None:
        data = client.post(
            "/api/query",
            json={"sql": "SELECT dept, COUNT(*) AS n FROM emp GROUP BY dept"},
        ).json()
        assert data["ok"] is True
        assert data["rows"] == [["Eng", 2]]

    def test_query_error(self, client: TestClient) -> None:
        res = client.post("/api/query", json={"sql": "SELECT * FROM nope"})
        assert res.status_code == 400
        assert res.json()["ok"] is False

    def test_explain(self, client: TestClient) -> None:
        data = client.post(
            "/api/explain",
            json={"sql": "SELECT name FROM emp WHERE salary > 100 AND 1 = 1"},
        ).json()
        assert data["ok"] is True
        assert "Scan" in data["optimized"]
        assert "optimized" in data and "original" in data

    def test_explain_error(self, client: TestClient) -> None:
        res = client.post("/api/explain", json={"sql": "SELECT bogus("})
        assert res.status_code == 400

    def test_index_served(self, client: TestClient) -> None:
        res = client.get("/")
        assert res.status_code == 200
        assert "prism" in res.text.lower()


class TestDefaultDatabase:
    def test_seeds_bundled_data(self) -> None:
        db = default_database()
        assert "employees" in db.catalog
        assert "departments" in db.catalog

    def test_uses_given_csv(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "t.csv"
        path.write_text("a\n1\n2")
        db = default_database([str(path)])
        assert "t" in db.catalog
