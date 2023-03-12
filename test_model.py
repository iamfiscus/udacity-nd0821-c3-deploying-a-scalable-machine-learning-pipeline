import pytest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, load

data = pd.read_csv('./data/census.csv')
model = load('./model/random_forest.pkl')

train, test = train_test_split(data, test_size=0.20)

"""
This code is used to process data for a machine learning model. 
The variable cat_features is an array of strings containing the names of the categorical features in the dataset. 
The function process_data is called twice, 
once with the argument training set to True and once with the argument training set to False. 
In the first call, X_train and y_train are assigned the return values of process_data, as well as encoder and lb. 
In the second call, X_test and y_test are assigned the return values of process_data, 
while encoder and lb are passed as arguments to ensure that they are used in processing the test data.
"""
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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)


# TEsts
def test_train_model():
    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)


def test_inference():
    preds = inference(model, X_test)

    assert len(preds) == len(y_test)
    assert isinstance(preds, np.ndarray)


def test_compute_model_metrics():
    preds = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
