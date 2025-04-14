import pandas as pd
import numpy as np
import os

from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from feature_engineering import *

def predict_eval(model, train, train_features, name):
    y_train_pred = model.predict(train[train_features])
    rmse = root_mean_squared_error(train.log_trip_duration, y_train_pred)
    r2 = r2_score(train.log_trip_duration, y_train_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")


def approach1(train, test): 
    numeric_features = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    categorical_features = ['dayofweek', 'month', 'hour', 'dayofyear', 'passenger_count']
    train_features = categorical_features + numeric_features

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ('scaling', StandardScaler(), numeric_features)
        ]
        , remainder = 'passthrough'
    )

    pipeline = Pipeline(steps=[
        ('ohe', column_transformer),
        ('regression', Ridge())
    ])

    model = pipeline.fit(train[train_features], train.log_trip_duration)
    predict_eval(model, train, train_features, "train")
    predict_eval(model, test, train_features, "test")

def approach2(train, test):
    numeric_features = [
                        'haversine_distance',
                        'manhattan_distance',
                        'speed_kmph',
                        'pickup_latitude',
                        'pickup_longitude',
                        'dropoff_latitude',
                        'dropoff_longitude', 
                        'pickup_cluster',
                        'dropoff_cluster',
                        'direction',
                        'distance_speed_combo',
                        ]
    categorical_features = [
                        'dayofweek',
                        'month', 
                        'hour', 
                        'dayofyear', 
                        'passenger_count',
                        'vendor_day_combo',
                        'vendor_month_combo',
                        'is_weekend',
                        'vendor',
                        ]
    train_features = categorical_features + numeric_features

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ('scaling', StandardScaler(), numeric_features)
        ]
        , remainder = 'passthrough'
    )

    pipeline = Pipeline(steps=[
        ('ohe', column_transformer),
        ('regression', Ridge(alpha=1))
    ])

    model = pipeline.fit(train[train_features], train.log_trip_duration)
    predict_eval(model, train, train_features, "train")
    predict_eval(model, test, train_features, "test")


def prepare_data(data):
    data.drop(columns=['id'], inplace=True)

    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    data['dayofweek'] = data.pickup_datetime.dt.dayofweek
    data['month'] = data.pickup_datetime.dt.month
    data['hour'] = data.pickup_datetime.dt.hour
    data['dayofyear'] = data.pickup_datetime.dt.dayofyear
    data['is_weekend'] = data['dayofweek'].isin([5, 6]).astype(int)
    data['vendor_day_combo'] = data['vendor_id'].astype(str) + '_' + data['dayofweek'].astype(str)
    data['vendor_month_combo'] = data['vendor_id'].astype(str) + '_' + data['month'].astype(str)

    # train['is_peak_day'] = train['dayofweek'].isin([0,1,2,3,4]).astype(int)
    data['vendor'] = data.vendor_id -1 

    haversine(data)
    manhattan(data)
    direction(data)
    clustering(data)
    speed(data)
    distance_speed_combo(data)

    if 'trip_duration' in data.columns:
            data['log_trip_duration'] = np.log1p(data.trip_duration)


if __name__ == '__main__':
    root_dir = 'project-1-trip-duration-prediction'
    # sample
    # print("\n" + "="*50)
    # print("Sample Results")
    # train = pd.read_csv(os.path.join(root_dir, 'data/split_sample/train.csv'))
    # test = pd.read_csv(os.path.join(root_dir, 'data/split_sample/val.csv'))
    # final_test = pd.read_csv(os.path.join(root_dir, 'data/split_sample/test.csv'))

    # full
    print("="*50)
    print("Full Results")
    train = pd.read_csv(os.path.join(root_dir, 'data/split/train.csv'))
    test = pd.read_csv(os.path.join(root_dir, 'data/split/val.csv'))
    final_test = pd.read_csv(os.path.join(root_dir, 'data/split/test.csv'))

    prepare_data(train)
    prepare_data(test)
    prepare_data(final_test)

    print("\n" + "="*50)
    print("Approach 1")
    approach1(train, test)
    
    print("\n" + "="*50)
    print("Approach 2") # my solution
    approach2(train, test)
    approach2(train, final_test)