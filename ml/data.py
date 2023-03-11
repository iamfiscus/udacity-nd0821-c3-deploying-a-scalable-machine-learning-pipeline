import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """
    This function takes in 
        a dataframe (X), 
        a list of categorical features, 
        a label (if applicable), 
        whether or not it is training, 
        an encoder, 
        and a label binarizer. 

    If a label is provided, the function will separate the dataframe into X and y. 
    It will then separate the dataframe into continuous and categorical features. 
    If it is training, it will fit the encoder and label binarizer to the data. 
    Otherwise, it will transform the data with the existing encoder and label binarizer. 

    Finally, it will concatenate the continuous and 
    categorical features together and 
    return X, y, encoder, and lb.
    """
    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
