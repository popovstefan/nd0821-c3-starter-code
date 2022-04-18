"""
Evaluate the model performance on different slices of data

author Stefan Popov
date Apr 2022
"""
import os.path

from joblib import load

import train_model
from ml.data import process_data
from ml.model import compute_model_metrics

trained_model = load(os.path.join(os.path.dirname(__file__),
                                  "../model/model.joblib"))
encoder = load(os.path.join(os.path.dirname(__file__),
                            "../model/encoder.joblib"))
lb = load(os.path.join(os.path.dirname(__file__),
                       "../model/lb.joblib"))

slice_values = []

for cat_ftr in train_model.cat_features:
    for value in train_model.test[cat_ftr].unique():
        df_temp = train_model.test[train_model.test[cat_ftr] == value]
        X_test, y_test, _, _ = process_data(
            df_temp,
            categorical_features=train_model.cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            training=False
        )
        y_pred = trained_model.predict(X_test)
        prc, rcl, fb = compute_model_metrics(y_test, y_pred)
        line = "[%s = %s] Precision: %s Recall: %s FBeta: %s" \
               % (cat_ftr, value, prc, rcl, fb)
        slice_values.append(line)

# write out the slice performance values
fn = os.path.join(os.path.dirname(__file__),
                  '../model/slice_output.txt')
with open(fn, 'w') as f:
    f.write("\n".join(slice_values))
