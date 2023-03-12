from fastapi.testclient import TestClient
import pandas as pd

# Import
from main import app

# Instantiate FastAPI TestClient
client = TestClient(app)

data_columns = ["age",
                "workclass",
                "fnlgt",
                "education",
                "education-num",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "capital-gain",
                "capital-loss",
                "hours-per-week",
                "native-country"]


# Write tests
def test_get_greet_user():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "data": "Hello World. This app is a FastAPI for Project 3 of Udacity's ML for DevOps Nanodegree"
    }


def test_post_model_predict_higher():
    input_data = pd.DataFrame(
        [[56, "Local-gov", 216851, "Bachelors", 13, "Married-civ-spouse",
            "Tech-support", "Husband", "White", "Male", 0, 0, 40, "United-States"]],
        columns=data_columns
    )
    r = client.post("/inference", json=input_data.iloc[0, :].to_dict())

    assert r.status_code == 200
    assert r.json() == {"result": {'0': ">50k"}}


def test_post_model_predict_lower():
    input_data = pd.DataFrame(
        [[38, "Private", 215646, "HS-grad", 9, "Divorced", "Handlers-cleaners",
            "Not-in-family", "White", "Male", 0, 0, 40, "United-States"]],
        columns=data_columns
    )
    r = client.post("/inference", json=input_data.iloc[0, :].to_dict())

    assert r.status_code == 200
    assert r.json() == {"result": {'0': "<=50k"}}
