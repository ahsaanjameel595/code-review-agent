import pytest
from fastapi.testclient import TestClient

from main import app, reset_store


@pytest.fixture(autouse=True)
def clear_store():
    reset_store()
    yield
    reset_store()


@pytest.fixture
def client():
    return TestClient(app)


def test_list_empty(client):
    response = client.get("/todos")

    assert response.status_code == 200
    assert response.json() == []


def test_create_todo(client):
    response = client.post("/todos", json={"title": "Learn LangGraph"})

    assert response.status_code == 201
    assert response.json() == {"id": 1, "title": "Learn LangGraph", "done": False}


def test_get_todo(client):
    created = client.post("/todos", json={"title": "Write tests"}).json()

    response = client.get(f"/todos/{created['id']}")

    assert response.status_code == 200
    assert response.json() == created


def test_get_missing_todo_returns_404(client):
    assert client.get("/todos/100").status_code == 404


def test_update_todo(client):
    created = client.post("/todos", json={"title": "Review code"}).json()

    response = client.put(
        f"/todos/{created['id']}",
        json={"title": "Review code carefully", "done": True},
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": created["id"],
        "title": "Review code carefully",
        "done": True,
    }


def test_delete_todo(client):
    created = client.post("/todos", json={"title": "Delete me"}).json()

    response = client.delete(f"/todos/{created['id']}")

    assert response.status_code == 204
    assert client.get(f"/todos/{created['id']}").status_code == 404


def test_empty_title_is_rejected(client):
    assert client.post("/todos", json={"title": ""}).status_code == 422
