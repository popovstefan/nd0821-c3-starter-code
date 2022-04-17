import os

import joblib
import pandas as pd
import pytest as pytest

from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference
from starter.starter.train_model import cat_features


@pytest.fixture
def df() -> pd.DataFrame:
    fp = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     "starter/data/census_clean.csv")
    )
    return pd.read_csv(fp)


def test_process_data(df):
    """
    Check that splits have the same number of rows for X and y
    """
    fp = os.path.join(os.path.dirname(__file__),
                      "starter/model/encoder.joblib")
    encoder = joblib.load(fp)
    fp = os.path.join(os.path.dirname(__file__),
                      "starter/model/lb.joblib")
    lb = joblib.load(fp)

    X, y, _, _ = process_data(df,
                              categorical_features=cat_features,
                              label="salary",
                              encoder=encoder,
                              lb=lb,
                              training=False)

    assert len(X) == len(y)


def test_encoder_and_lb_params(df):
    """
    Check encoder and label-binarizer parameters
    in training and inference setting.
    """
    fp = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     "starter/model/encoder.joblib")
    )
    encoder_test = joblib.load(fp)
    fp = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     "starter/model/lb.joblib")
    )
    lb_test = joblib.load(fp)

    _, _, encoder, lb = process_data(df,
                                     categorical_features=cat_features,
                                     label="salary",
                                     training=True)

    _ = process_data(df,
                     categorical_features=cat_features,
                     label="salary",
                     encoder=encoder_test,
                     lb=lb_test,
                     training=False)

    assert encoder.get_params() == encoder_test.get_params()
    assert lb.get_params() == lb_test.get_params()


def test_inference_above():
    """
    Check inference performance (sample data taken from the eda.ipynb)
    """
    fp = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     "starter/model/model.joblib")
    )
    model = joblib.load(fp)
    fp = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     "starter/model/encoder.joblib")
    )
    encoder = joblib.load(fp)
    fp = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     "starter/model/lb.joblib")
    )
    lb = joblib.load(fp)

    test_df = pd.DataFrame(data=[[
        55,
        "Private",
        "Some-college",
        "Married-civ-spouse",
        "Exec-managerial",
        "Husband",
        "White",
        "Male",
        40,
        "United-States"
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

    X, _, _, _ = process_data(
        test_df,
        categorical_features=cat_features,
        encoder=encoder, lb=lb, training=False)
    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == ">50K"


def test_inference_below():
    """
    Check inference performance (sample data taken from the eda.ipynb)
    """
    fp = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     "starter/model/model.joblib")
    )
    model = joblib.load(fp)
    fp = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     "starter/model/encoder.joblib")
    )
    encoder = joblib.load(fp)
    fp = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     "starter/model/lb.joblib")
    )
    lb = joblib.load(fp)

    test_df = pd.DataFrame(data=[[
        45,
        "State-gov",
        "HS-grad",
        "Divorced",
        "Protective-serv",
        "Unmarried",
        "Black",
        "Female",
        40,
        "United-States"
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

    X, _, _, _ = process_data(test_df,
                              categorical_features=cat_features,
                              encoder=encoder,
                              lb=lb,
                              training=False)
    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == "<=50K"
