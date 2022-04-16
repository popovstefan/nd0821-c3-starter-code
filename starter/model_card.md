# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is a GradientBoostingClassifier built with the default parameters from the `scikit-learn` library 
(we only explicitly set the number of base estimators to 100).

The model has been developed by Stefan Popov.

## Intended Use

The model should be able to predict whether the salary of a person exceeds a certain
threshold based on social attributes.

## Training Data

The data has been downloaded from this [link](https://archive.ics.uci.edu/ml/datasets/census+income). 

Model training is performed using a random sample of size equal to 80% of this data.

## Evaluation Data

The data has been downloaded from this [link](https://archive.ics.uci.edu/ml/datasets/census+income). 

Model evaluation is performed using a random sample of size equal to 20% of this data (and not including any training data).

## Metrics

You can view these in `starter/model/slice_performances.txt`

## Ethical Considerations

Data set contains sensitive information, such as gender, race and country of origin. 
This may cause the model to potentially discriminate against certain group of people.
It should be used with caution.

## Caveats and Recommendations

Maybe we can add additional attributes, such as health (which could impact the person's earnings), or years of experience in the field.