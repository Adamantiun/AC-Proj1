
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
import numpy as np
import warnings
import os

# Set the warning filter to "ignore"
warnings.filterwarnings("ignore", category=UserWarning)

def save_model(model, X_train, X_test, y_train, y_test, fname):
    # Define directory name
    dirname = 'models'

    # Create directory if it doesn't exist
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Save the model and data
    joblib.dump({'model': model, 'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}, os.path.join(dirname, fname))

def load_model(fname):
    dirname = 'models'
    model = joblib.load(os.path.join(dirname, fname))
    return model

def train_model(data, model_obj, fname, test_size=0.2):
    X = data.values
    y = np.arange(X.shape[0])  # Create an array of indices
    Le = LabelEncoder()
    for i in range(X.shape[1]):
        X[:, i] = Le.fit_transform(X[:, i])

    # train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)  # Split the data into training and testing fields
    model_obj.fit(X_train, y_train)

    # Save trained model and data split to file, so it can be used later for testing purposes
    save_model(model_obj, X_train, X_test, y_train, y_test, fname)
