from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from ml.data import process_data

import pickle


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    This function takes two parameters, X_train and y_train, and
    uses them to train a RandomForestClassifier model. The model is then returned.
    X_train is expected to be a 2D array-like object containing the training data features,
    while y_train is expected to be a 1D array-like object containing the target values for each sample in X_train.
    """

    random_forest_classifier = RandomForestClassifier()
    random_forest_classifier.fit(X_train, y_train)

    return random_forest_classifier


def save_model(model, filepath):
    """
    save_model() is a function that takes two parameters: model and filepath.
    It opens the filepath in write binary mode and uses pickle to dump the model into the file.
    """

    with open(filepath, 'wb') as model_file:
        pickle.dump(model, model_file)


def load(filepath):
    """
    This function loads a file from the given filepath using the pickle library.
    It opens the file in read binary mode and loads it into a model variable.
    The model is then returned.
    """
    with open(filepath, 'rb') as model_file:
        model = pickle.load(model_file)
    return model


def compute_model_metrics(y, preds):
    """
    This function computes the precision, recall, and fbeta scores for a given model.
    The function takes two arguments: y (the true labels) and preds (the predicted labels).
    The beta parameter is set to 1, and the zero_division parameter is set to 1.
    The function returns the precision, recall, and fbeta scores as a tuple.

    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):

    return model.predict(X)
