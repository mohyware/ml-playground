import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
import os

RANDOM_STATE = 17
np.random.seed(RANDOM_STATE)

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "data2_200x30.csv")

# Load data
df = pd.read_csv(file_path) 
df = df.fillna(df.median())
data = df.to_numpy()

X = data[:, :-1]
t = data[:, -1].reshape(-1, 1)
# Split data
X_train, X_val, t_train, t_val = train_test_split(X, t, test_size=0.50,shuffle=False, random_state=None)
# Scale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit & transform on train data
X_val_scaled = scaler.transform(X_val)  # Only transform on validation data

def linear_regression(X, t):
    model = LinearRegression(fit_intercept = True)
    model.fit(X, t)
    return model

def train(X_train_scaled,t_train):
    result = linear_regression(X_train_scaled, t_train)
    avg_abs_weght = abs(result.coef_).mean()
    #print("Model coefficients: ", result.coef_)
    print("Intercept:", result.intercept_)    # Bias term
    #print("Score (R²):", result.score(X_train, t_train))  # Model performance
    print("Average absolute weight: ", avg_abs_weght)
    return result

def eval(result,X_train_scaled,X_val_scaled):
    # Predictions
    t_train_pred = result.predict(X_train_scaled)
    t_val_pred = result.predict(X_val_scaled)

    # Compute errors
    train_error = mean_squared_error(t_train, t_train_pred)
    val_error = mean_squared_error(t_val, t_val_pred)
    print("Error of train:", math.sqrt(train_error))
    print("Error of val:", math.sqrt(val_error))

def monomials_poly_features(X, degree):
    '''
    For each feature xi, creates: xi^1, xi^2, xi^3...xi^degree without any cross features (xi * xj)
    '''
    assert degree > 0

    if degree == 1:
        return X

    examples = []
    # ToDo make it faster/pythonic? How?
    for example in X:
        example_features = []
        for feature in example:
            cur = 1
            feats = []
            for deg in range(degree):
                cur *= feature
                feats.append(cur)
            example_features.extend(feats)
        examples.append(np.array(example_features))

    return np.vstack(examples)
