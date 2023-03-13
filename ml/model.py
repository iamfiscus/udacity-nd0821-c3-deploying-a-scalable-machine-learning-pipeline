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


def compute_model_performance_on_slices(data, label, features, cat_features, model, encoder, lb):
    """
    This function computes the performance of a given model on slices of data. It takes in the following parameters:
    - data: a pandas DataFrame containing the data to be processed
    - label: the label column of the DataFrame
    - features: a list of features to slice by
    - cat_features: a list of categorical features in the DataFrame
    - model: a trained machine learning model
    - encoder: an encoder object used to encode categorical variables
    - lb: an instance of sklearn's LabelBinarizer object used to binarize labels

    The function first initializes an empty string and an empty list for storing performance metrics.
    It then loops through each feature in the list 'features' and its unique values, and processes the data using process_data().
    The model is then used to make predictions on this slice of data, and compute_model_metrics() is used to calculate
    precision, recall, and fbeta scores.

    These scores are printed out and stored in all_performance string and model_performance list.

    Finally, a pandas DataFrame is created from the model_performance list, which contains columns for constant feature
    name, value, precision, recall, and fbeta scores.
    This DataFrame is written to a file called 'slice_output.txt', and returned by the function.
    """
    all_performance = ''
    model_performance = []
    for feature in features:
        values = data[feature].unique()
        for value in values:
            data_to_test = data[data[feature] == value]
            print(cat_features)
            X_slice, y_slice, _, _ = process_data(
                data_to_test, categorical_features=cat_features, label=label, training=False, encoder=encoder, lb=lb
            )
            preds = inference(model, X_slice)

            precision, recall, fbeta = compute_model_metrics(y_slice, preds)

            run_performance = f'''
            Feature {feature}=={value}:
            \tprecision: {precision}'
            \trecall: {recall}
            \tbeta: {fbeta}

            '''
            print(run_performance)

            all_performance += run_performance
            model_performance.append(
                [feature, value, precision, recall, fbeta])
    model_performance_df = pd.DataFrame(model_performance, columns=[
                                        'constant_column', 'value', 'precision', 'recall', 'fbeta'])

    with open('slice_output.txt', 'w') as performance_file:
        performance_file.write(all_performance)

    return model_performance_df


def inference(model, X):

    return model.predict(X)
