import pytest
from tag.modelloFlask import app as flask_app

@pytest.fixture()
def client():
    flask_app.config.update({"TESTING": True})
    with flask_app.test_client() as client:
        yield client


def test_predict(client):
    response = client.post("/predict", json={"hyper_param": 5})
    assert response.status_code == 200
    
    data = response.get_json()
    print(f"Response data: {data}")

    assert "stipendio_previsto" in data
    assert isinstance(data["stipendio_previsto"], float)
    assert "latency_seconds" in data