# Script to train machine learning model.
import os.path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .ml.data import process_data
from .ml.model import train_model, compute_model_metrics

data_filepath = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "../data/census_clean.csv")
)

train, test = train_test_split(pd.read_csv(data_filepath), test_size=0.2)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# pre-process data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
# train model
model = train_model(X_train, y_train)

# score model
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    encoder=encoder,
    lb=lb,
    training=False
)
y_pred = model.predict(X_test)
prc, rcl, fb = compute_model_metrics(y_test, y_pred)
line = "Precision: %s Recall: %s FBeta: %s" % (prc, rcl, fb)
fn = os.path.join(os.path.dirname(__file__),
                  '../model/test_evaluation.txt')
with open(fn, 'w') as f:
    f.write(line)

# save output
fp = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                  "../model/model.joblib"))
joblib.dump(model, fp)
fp = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                  "../model/encoder.joblib"))
joblib.dump(encoder, fp)
fp = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                  "../model/lb.joblib"))
joblib.dump(lb, fp)
