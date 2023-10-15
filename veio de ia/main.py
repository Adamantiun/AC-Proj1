import data_manip
import os
import model_test
import warnings
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from enum import Enum
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from pycaret.classification import *
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class ModelType(Enum):
    svm = 1
    logreg = 2
    graboo = 3
    MLPR = 4

# Set the warning filter to "ignore"
warnings.filterwarnings("ignore", category=UserWarning)

model_map = {
    ModelType.svm: (SVC(), 'svm.sav'),
    ModelType.logreg: (LogisticRegression(), 'logreg.sav'),
    ModelType.graboo: (GradientBoostingRegressor(), 'gradientbooster.sav'),
    ModelType.MLPR: (MLPRegressor(), 'MLPRegressor.sav'),
}

current_dir = os.path.dirname(os.path.abspath(__file__))
filename = "airlines_delay_original.csv"
data = pd.read_csv(os.path.join(current_dir, filename))
data_encoded = data.copy()

# Combine "AirportFrom" and "AirportTo" columns to create a unified set of airports
airports = pd.concat([data_encoded['AirportFrom'], data_encoded['AirportTo']])

# Perform label encoding on the combined set of airports
label_encoder_airports = LabelEncoder()
encoded_airports = label_encoder_airports.fit_transform(airports)

# Perform label encoding on the "Airline" column
label_encoder_airline = LabelEncoder()
data_encoded['Airline'] = label_encoder_airline.fit_transform(data_encoded['Airline'])

# Update "AirportFrom" and "AirportTo" columns with the encoded values
data_encoded['AirportFrom'] = encoded_airports[:len(data_encoded)]
data_encoded['AirportTo'] = encoded_airports[len(data_encoded):]

# Remove the "Flight" column
data_encoded = data_encoded.drop('Flight', axis=1)

# Save the encoded data to a CSV file
data_encoded.to_csv('airlines_delay_encoded.csv', index=False)
#cat_features = ['Flight','Time','Length','Airline','AirportFrom','AirportTo','DayOfWeek','Class']
#fig , ax = plt.subplots(8,1,figsize = (40,40))     # set up 2 x 2 frame count plot with figsize 10 x 10
#for i , subplots in zip (cat_features, ax.flatten()):
#    sns.countplot(data=data,x=data[i],hue = data['Class'],ax = subplots, palette = 'BuPu')

X = data_encoded.drop('Class', axis = 1)
y = data_encoded['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

s = setup(data=data_encoded, target='Class', session_id=123, normalize=True)
compare_models()
#data = pd.read_csv("airlines_delay.csv")
#data = pd.read_csv("iris-data.csv")

# train and test svm model
#data_manip.train_model(data, model_map[ModelType.svm][0], model_map[ModelType.svm][1])
#model_test.test_model(model_map[ModelType.svm][1])

# train and test logical regression model
#data_manip.train_model(data, model_map[ModelType.logreg][0], model_map[ModelType.logreg][1])
#model_test.test_model(model_map[ModelType.logreg][1])

# train and test gradient booster regressor model
#data_manip.train_model(data, model_map[ModelType.graboo][0], model_map[ModelType.graboo][1])
#model_test.test_model(model_map[ModelType.graboo][1])

# train and test MLPRegressor model
#data_manip.train_model(data, model_map[ModelType.MLPR][0], model_map[ModelType.MLPR][1])
#model_test.test_model(model_map[ModelType.MLPR][1])
