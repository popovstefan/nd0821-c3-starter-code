# Exposing a machine learning model through API on Heroku


Udacity project about creating an ML pipeline to train a and evaluate a model and then publish it with through an API on Heroku.

## EDA

An EDA Jupyter notebook is available [here](starter/eda.ipynb).

## Githooks and actions

We have configured a `pre-commit` hook to run `flake8` locally before allowing the commit to occur.
Unless the command passes, the commit won't pass.

On each push to master, we have configured to run `pytest` and `flake8`. Both commands
should pass.

## Model details and performance

There is a model card containing all the information related to the model [here](model_card.md).