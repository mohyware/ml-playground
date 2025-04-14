import numpy as np

def haversine(data):
    from haversine import haversine
    data['haversine_distance'] = data.apply(lambda x: haversine((x['pickup_latitude'],  x['pickup_longitude']), 
                                                            (x['dropoff_latitude'], x['dropoff_longitude'])), axis=1)

def manhattan(data):
    data["manhattan_distance"] = (
    np.abs(data["dropoff_longitude"] - data["pickup_longitude"]) +
    np.abs(data["dropoff_latitude"] - data["pickup_latitude"])
)

def clustering(data):
    from sklearn.cluster import MiniBatchKMeans
    pickup_coords = data[['pickup_latitude', 'pickup_longitude']].values
    dropoff_coords = data[['dropoff_latitude', 'dropoff_longitude']].values

    coords = np.vstack((pickup_coords, dropoff_coords))

    kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords)

    data['pickup_cluster']  = kmeans.predict(pickup_coords)
    data['dropoff_cluster'] = kmeans.predict(dropoff_coords)

def speed(data):
    data['speed_kmph'] = data.apply(lambda x: x['haversine_distance'] / x.trip_duration / 3600 if x.trip_duration > 0 else np.nan, axis=1)

def ft_degree(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371 
    lng_delta_rad = np.radians(lng2 - lng1)   
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x)) 

def direction(data):
    data['direction'] = ft_degree(data['pickup_latitude'].values,
                                data['pickup_longitude'].values,
                                data['dropoff_latitude'].values,
                                data['dropoff_longitude'].values)

def distance_speed_combo(data):
    data['distance_speed_combo'] = data.apply(
        lambda row: row['haversine_distance'] / row['speed_kmph']
        if row['speed_kmph'] > 0 else 0,
        axis=1
    )