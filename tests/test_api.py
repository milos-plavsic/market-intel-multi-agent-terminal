from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health() -> None:
    """Execute the test health routine."""
    assert client.get("/health").status_code == 200


def test_brief() -> None:
    """Execute the test brief routine."""
    r = client.post("/v1/brief", json={"ticker": "aapl"})
    assert r.status_code == 200
    assert r.json()["ticker"] == "AAPL"
