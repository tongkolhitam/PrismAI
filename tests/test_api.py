import pytest
from fastapi.testclient import TestClient
from prismai.api.server import app

client = TestClient(app)


def test_health():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_stats():
    r = client.get("/api/v1/stats")
    assert r.status_code == 200


def test_history():
    r = client.get("/api/v1/history")
    assert r.status_code == 200
