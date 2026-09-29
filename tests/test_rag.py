from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ask_endpoint():
    response = client.post(
        "/ask",
        json={
            "question": "What are the key components of an Agentic AI system?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"]
    assert data["answer"]
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) > 0


def test_empty_question():
    response = client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 422