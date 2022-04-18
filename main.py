"""
The API for the project

author Stefan Popov
date Apr 2022
"""
import os

import joblib
import pandas as pd
from fastapi import FastAPI
from typing import Literal
from starter.starter.ml.data import process_data
from starter.starter.ml import model
from pydantic.main import BaseModel

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull -R") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()


class Person(BaseModel):
    class Config:
        schema_extra = {
            "example": {
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
            }
        }

    age: int
    workclass: Literal['State-gov',
                       'Self-emp-not-inc',
                       'Private',
                       'Federal-gov',
                       'Local-gov',
                       'Self-emp-inc',
                       'Without-pay']
    education: Literal['Bachelors',
                       'HS-grad',
                       '11th',
                       'Masters',
                       '9th',
                       'Some-college',
                       'Assoc-acdm',
                       '7th-8th',
                       'Doctorate',
                       'Assoc-voc',
                       'Prof-school',
                       '5th-6th',
                       '10th',
                       'Preschool',
                       '12th',
                       '1st-4th']
    maritalStatus: Literal['Never-married',
                           'Married-civ-spouse',
                           'Divorced',
                           'Married-spouse-absent',
                           'Separated',
                           'Married-AF-spouse',
                           'Widowed']
    occupation: Literal['Adm-clerical',
                        'Exec-managerial',
                        'Handlers-cleaners',
                        'Prof-specialty',
                        'Other-service',
                        'Sales',
                        'Transport-moving',
                        'Farming-fishing',
                        'Machine-op-inspct',
                        'Tech-support',
                        'Craft-repair',
                        'Protective-serv',
                        'Armed-Forces',
                        'Priv-house-serv']
    relationship: Literal['Not-in-family',
                          'Husband',
                          'Wife',
                          'Own-child',
                          'Unmarried',
                          'Other-relative']
    race: Literal['White',
                  'Black',
                  'Asian-Pac-Islander',
                  'Amer-Indian-Eskimo',
                  'Other']
    sex: Literal['Male', 'Female']
    hoursPerWeek: int
    nativeCountry: Literal['United-States',
                           'Cuba',
                           'Jamaica',
                           'India',
                           'Mexico',
                           'Puerto-Rico',
                           'Honduras',
                           'England',
                           'Canada',
                           'Germany',
                           'Iran',
                           'Philippines',
                           'Poland',
                           'Columbia',
                           'Cambodia',
                           'Thailand',
                           'Ecuador',
                           'Laos',
                           'Taiwan',
                           'Haiti',
                           'Portugal',
                           'Dominican-Republic',
                           'El-Salvador',
                           'France',
                           'Guatemala',
                           'Italy',
                           'China',
                           'South',
                           'Japan',
                           'Yugoslavia',
                           'Peru',
                           'Outlying-US(Guam-USVI-etc)',
                           'Scotland',
                           'Trinadad&Tobago',
                           'Greece',
                           'Nicaragua',
                           'Vietnam',
                           'Hong',
                           'Ireland',
                           'Hungary',
                           'Holand-Netherlands']


@app.get("/")
async def get_items():
    return {"message": "Greetings Earthling!"}


@app.on_event("startup")
async def startup_event():
    global clf, encoder, lb
    clf = joblib.load(os.path.join(os.path.dirname(__file__),
                                   "starter/model/model.joblib"))
    encoder = joblib.load(os.path.join(os.path.dirname(__file__),
                                       "starter/model/encoder.joblib"))
    lb = joblib.load(os.path.join(os.path.dirname(__file__),
                                  "starter/model/lb.joblib"))


@app.post("/inference")
async def inference(person: Person):
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
                              categorical_features=[
                                  "workclass",
                                  "education",
                                  "marital-status",
                                  "occupation",
                                  "relationship",
                                  "race",
                                  "sex",
                                  "native-country",
                              ],
                              encoder=encoder,
                              lb=lb,
                              training=False)
    pred = model.inference(clf, X)
    y = lb.inverse_transform(pred)[0]
    return {
        "prediction": y
    }
