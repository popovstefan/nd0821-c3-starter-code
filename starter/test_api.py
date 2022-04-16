import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_get(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Greetings Earthling!"}


def test_post_below(client):
    r = client.post("/inference", json={
        "age": 45,
        "workclass": "State-gov",
        "education": "HS-grad",
        "maritalStatus": "Divorced",
        "occupation": "Protective-serv",
        "relationship": "Unmarried",
        "race": "Black",
        "sex": "Female",
        "hoursPerWeek": 40,
        "nativeCountry": "United-States"
    })
    assert r.json() == {"prediction": "<=50K"}
    assert r.status_code == 200


def test_post_above(client):
    r = client.post("/inference", json={
        "age": 55,
        "workclass": "Private",
        "education": "Some-college",
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "hoursPerWeek": 40,
        "nativeCountry": "United-States"
    })
    assert r.json() == {"prediction": ">50K"}
    assert r.status_code == 200
