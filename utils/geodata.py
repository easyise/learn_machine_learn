import pandas as pd
import numpy as np
import geopandas as gpd

from shapely.geometry import LineString

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's surface.

    :param lat1: Latitude of the first point in decimal degrees
    :param lon1: Longitude of the first point in decimal degrees
    :param lat2: Latitude of the second point in decimal degrees
    :param lon2: Longitude of the second point in decimal degrees
    :return: Distance in meters
    """
    R = 6371000  # метры
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2

    return 2 * R * math.asin(math.sqrt(a))

# функция расчета расстояний, скоростей и ускорений
def recalculate_metrics(gdf_):
    """
    Recalculate distance, speed, and acceleration metrics for a GeoDataFrame.

    Distance is calculated using the haversine formula, speed is calculated as distance over time, and acceleration is calculated as the change in speed over time.

    Timedeltas are calculated in seconds, speed is converted to km/h, and acceleration is calculated in m/s² and g-forces.

    All data is calculated "on arrival", meaning that the first row will have NaN for distance, speed, and acceleration, which are then filled with 0.

    :param gdf_: GeoDataFrame with geometry and time columns
    :return: GeoDataFrame with additional columns for distance, speed, and acceleration
    """
    # Calculate distances in meters by haversine formula
    gdf_['prev_point'] = gdf_.geometry.shift(1)
    gdf_['dist'] = gdf_.apply(lambda row: haversine(
        row.geometry.y, row.geometry.x,
        row.prev_point.y if pd.notna(row.prev_point) else row.geometry.y,
        row.prev_point.x if pd.notna(row.prev_point) else row.geometry.x
    ), axis=1)

    # Calculate time differences in seconds
    gdf_['timedelta_s'] = gdf_['time'].diff().dt.total_seconds()
    gdf_['timedelta_s'] = gdf_['timedelta_s'].fillna(0)

    # Calculate speed in m/s
    gdf_['speed_kmh'] = gdf_['dist'] / gdf_['timedelta_s'] * 3.6
    gdf_['speed_kmh'] = gdf_['speed_kmh'].fillna(0)

    # Calculate acceleration in m/s² (change in speed over time)
    # gdf_['ds'] = gdf_['speed_kmh'].diff()
    gdf_['acc_m_per_s2'] = (gdf_['speed_kmh'] / 3.6).diff() / gdf_['timedelta_s']
    gdf_['acc_g'] = gdf_['acc_m_per_s2'] / 9.81

    gdf_[['acc_m_per_s2', 'acc_g']] = gdf_[['acc_m_per_s2', 'acc_g']].fillna(0)
    
    return gdf_




def get_with_segments(gdf):
    """
    Recalculate line segments for a GeoDataFrame.

    :param gdf: GeoDataFrame with geometry column
    :return: GeoDataFrame with additional column for line segments
    """
    gdf['prev_point'] = gdf.geometry.shift(1)
    gdf['segment'] = gdf.apply(lambda row: LineString([row.prev_point, row.geometry]) 
                               if pd.notna(row.prev_point) else None, axis=1)
    
    gdf_ret = gpd.GeoDataFrame(
        gdf, geometry='segment', crs=gdf.crs
    )

    return gdf_ret


def get_z_scores(gdf, z, cols=['speed_kmh', 'acc_m_per_s2']):
    """
    Calculate robust z-scores for specified columns in a GeoDataFrame using MAD (Median Absolute Deviation).

    :param gdf: GeoDataFrame with columns to calculate z-scores for
    :param z: Z-score threshold
    :param cols: List of columns to calculate z-scores for
    """
    ret_thres = tuple()
    for col in cols:
        v = gdf[col]
        median = v.median()
        mad = (v - median).abs().median()
        ret_thres += (median + z * 1.4826 * mad,)
        gdf[f'{col}_z'] = np.abs(0.6745 * (v - median) / mad)
        gdf[f'{col}_z_fail'] = gdf[f'{col}_z'] > z
        gdf[f'{col}_z'] = gdf[f'{col}_z'].fillna(0)
        gdf[f'{col}_z_fail'] = gdf[f'{col}_z_fail'].fillna(False)
    return (gdf,) + ret_thres


def recalculate_triangulation_metrics(gdf_):
    """
    Recalculate triangulation metrics for a GeoDataFrame.

    :param gdf_: GeoDataFrame with geometry and time columns
    :return: GeoDataFrame with additional columns for triangulation metrics
    """
    
    gdf_['next_point'] = gdf_.geometry.shift(-1)
    
    # базис треугольника (расстояние между предыдущей и следующей точками)
    gdf_['dist_base'] = gdf_.apply(lambda row: haversine(
        row.prev_point.y if pd.notna(row.prev_point) else row.geometry.y,
        row.prev_point.x if pd.notna(row.prev_point) else row.geometry.x,
        row.next_point.y if pd.notna(row.next_point) else row.geometry.y,
        row.next_point.x if pd.notna(row.next_point) else row.geometry.x
    ), axis=1)

    # Calculate speeds out and base in km/h
    gdf_['speed_out'] = (gdf_['dist'].shift(-1) / gdf_['timedelta_s'].shift(-1) * 3.6).fillna(0)
    gdf_['speed_base'] = (gdf_['dist_base'] / (gdf_['timedelta_s'] + gdf_['timedelta_s'].shift(-1)) * 3.6).fillna(0)

    # Коэффициент различия между скоростями в основании и по сторонам треугольника. 
    # Если скорость в основании значительно ниже, чем по сторонам, то точка может быть выбросом.
    # Принимается значение менее 0.8 как подозрительное.
    max_speed = gdf_[['speed_kmh', 'speed_out']].max(axis=1)
    gdf_['triang_diff'] = (gdf_['speed_base'] / max_speed).fillna(1)

    return gdf_


def clustering_features(gdf_, include=['time', 'dist', 'speed_kmh','acc_m_per_s2']):
    gdf_ = gdf_.copy()
    
    # подбираем CRS с максимально подходящей проекцией
    utm_crs = gdf_.estimate_utm_crs()
    gdf_utm = gdf_.to_crs(utm_crs)

    # формируем матрицу признаков из метрических координат + времени + скорости
    coords_x = gdf_utm.geometry.x.to_numpy()
    coords_y = gdf_utm.geometry.y.to_numpy()
    list_of_features = [coords_x, coords_y]
    if 'time' in include:
        time_s = (gdf_.time - gdf_.time.iloc[0]).dt.total_seconds().to_numpy()
        list_of_features.append(time_s)
    if 'dist' in include:
        dist_m = gdf_['dist'].to_numpy()
        list_of_features.append(dist_m)
    if 'speed_kmh' in include:
        speed_kmh = gdf_['speed_kmh'].to_numpy()
        list_of_features.append(speed_kmh)
    if 'acc_m_per_s2' in include:
        acc_m_per_s2 = gdf_['acc_m_per_s2'].to_numpy()
        list_of_features.append(acc_m_per_s2)

    X = np.column_stack(list_of_features)

    # стандартизируем признаки
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


def choose_eps_knee(X, k=10, quantile=0.98):
    """
    X: (N,2) координаты в метрах (например, UTM)
    k: min_samples
    Возвращает eps по "колену" (упрощённо: высокий квантиль k-distance).
    """
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dists, _ = nn.kneighbors(X)
    kdist = np.sort(dists[:, -1])  # расстояние до k-го соседа

    # Простой устойчивый автопорог: почти-колено ≈ верхний квантиль плотной зоны
    eps = float(np.quantile(kdist, quantile))
    return eps, kdist


def get_cluster_labels(gdf, min_samples=600, include=['time', 'dist', 'speed_kmh']):
    
    recalculate_metrics(gdf)

    X = clustering_features(gdf, include=include)

    eps_m, kdist = choose_eps_knee(X, k=min_samples, quantile=0.98)

    labels = DBSCAN(eps=eps_m, 
                    min_samples=min_samples,
                    metric="euclidean").fit_predict(X)
    return labels, eps_m