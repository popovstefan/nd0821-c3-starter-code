"""
The API for the project

author Stefan Popov
date Apr 2022
"""
import os

import joblib
import pandas as pd
from fastapi import FastAPI

from person_model import Person
from starter import train_model
from starter.ml.data import process_data
from starter.ml import model


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()


@app.get("/")
async def get_items():
    return {"message": "Greetings Earthling!"}


@app.post("/inference")
async def inference(person: Person) -> dict[str, float]:
    clf = joblib.load(os.path.join(os.path.dirname(__file__), "model/model.joblib"))
    encoder = joblib.load(os.path.join(os.path.dirname(__file__), "model/encoder.joblib"))
    lb = joblib.load(os.path.join(os.path.dirname(__file__), "model/lb.joblib"))

    test_x = pd.DataFrame(data=[[
        person.age,
        person.workclass,
        person.education,
        person.maritalStatus,
        person.occupation,
        person.relationship,
        person.race,
        person.sex,
        person.hoursPerWeek,
        person.nativeCountry
    ]], columns=[
        "age",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
    ])

    X, _, _, _ = process_data(test_x,
                              categorical_features=train_model.cat_features,
                              encoder=encoder,
                              lb=lb,
                              training=False)
    pred = model.inference(clf, X)
    y = lb.inverse_transform(pred)[0]
    return {
        "prediction": y
    }
