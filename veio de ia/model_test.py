import data_manip
import joblib
import pandas as pd


def test_model(fname):
    # load test data
    data = data_manip.load_model(fname)
    X_test = data['X_test']
    y_test = data['y_test']
    model = data['model']
    # test model
    accuracy = model.score(X_test, y_test)
    print(f"{model} Accuracy: {accuracy:.2f}") # rounds the accuracy with 2 decimals
def use_model(fname, new_data):
    # load the trained model from file
    _, _, _, _, model = data_manip.load_model(fname)['model']

    # apply the model to make predictions
    predictions = model.predict(new_data)

    # return the predictions as a pandas Series
    return pd.Series(predictions)
