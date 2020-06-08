import pandas as pd
import numpy as np
import datetime
from keras import layers, models
from sklearn import preprocessing
from keras.utils import np_utils

#%% City range
X_CITY_MIN = 3750901.5068
X_CITY_MAX = 3770901.5068
Y_CITY_MIN = -19268905.6133
Y_CITY_MAX = -19208905.6133
X_MAX = 3777099.2656833767
X_MIN = 3740998.35481912
Y_MAX = -19382914.9809002
Y_MIN = -19042656.658487003

#%% Functions
def normalize_x(x):
    return (x - X_MIN )/(X_MAX - X_MIN)

def normalize_y(y):
    return (y - Y_MIN )/(Y_MAX - Y_MIN)

def normalize_time(t):
    return (t-1)/(32-1)

def time_to_range(x):
    if datetime.time(0, 0, 0) < x <= datetime.time(0, 30, 0):
        return 1
    elif datetime.time(0, 30, 0) < x <= datetime.time(1, 0, 0):
        return 2
    elif datetime.time(1, 0, 0) < x <= datetime.time(1, 30, 0):
        return 3
    elif datetime.time(1, 30, 0) < x <= datetime.time(2, 0, 0):
        return 4
    elif datetime.time(2, 0, 0) < x <= datetime.time(2, 30, 0):
        return 5
    elif datetime.time(2, 30, 0) < x <= datetime.time(3, 0, 0):
        return 6
    elif datetime.time(3, 0, 0) < x <= datetime.time(3, 30, 0):
        return 7
    elif datetime.time(3, 30, 0) < x <= datetime.time(4, 0, 0):
        return 8
    elif datetime.time(4, 0, 0) < x <= datetime.time(4, 30, 0):
        return 9
    elif datetime.time(4, 30, 0) < x <= datetime.time(5, 0, 0):
        return 10
    elif datetime.time(5, 0, 0) < x <= datetime.time(5, 30, 0):
        return 11
    elif datetime.time(5, 30, 0) < x <= datetime.time(6, 0, 0):
        return 12
    elif datetime.time(6, 0, 0) < x <= datetime.time(6, 30, 0):
        return 13
    elif datetime.time(6, 30, 0) < x <= datetime.time(7, 0, 0):
        return 14
    elif datetime.time(7, 0, 0) < x <= datetime.time(7, 30, 0):
        return 15
    elif datetime.time(7, 30, 0) < x <= datetime.time(8, 0, 0):
        return 16
    elif datetime.time(8, 0, 0) < x <= datetime.time(8, 30, 0):
        return 17
    elif datetime.time(8, 30, 0) < x <= datetime.time(9, 0, 0):
        return 18
    elif datetime.time(9, 0, 0) < x <= datetime.time(9, 30, 0):
        return 19
    elif datetime.time(9, 30, 0) < x <= datetime.time(10, 0, 0):
        return 20
    elif datetime.time(10, 0, 0) < x <= datetime.time(10, 30, 0):
        return 21
    elif datetime.time(10, 30, 0) < x <= datetime.time(11, 0, 0):
        return 22
    elif datetime.time(11, 0, 0) < x <= datetime.time(11, 30, 0):
        return 23
    elif datetime.time(11, 30, 0) < x <= datetime.time(12, 0, 0):
        return 24
    elif datetime.time(12, 0, 0) < x <= datetime.time(12, 30, 0):
        return 25
    elif datetime.time(12, 30, 0) < x <= datetime.time(13, 0, 0):
        return 26
    elif datetime.time(13, 0, 0) < x <= datetime.time(13, 30, 0):
        return 27
    elif datetime.time(13, 30, 0) < x <= datetime.time(14, 0, 0):
        return 28
    elif datetime.time(14, 0, 0) < x <= datetime.time(14, 30, 0):
        return 29
    elif datetime.time(14, 30, 0) < x <= datetime.time(15, 0, 0):
        return 30
    elif datetime.time(15, 0, 0) < x <= datetime.time(15, 30, 0):
        return 31
    elif datetime.time(15, 30, 0) < x <= datetime.time(16, 0, 0):
        return 32
    else:
        return 0

def check_exit(row):
    if X_CITY_MIN <= row['x_exit'] <= X_CITY_MAX and Y_CITY_MIN <= row['y_exit'] <= Y_CITY_MAX:
        return 1
    else:
        return 0

def check_entry(row):
    if X_CITY_MIN <= row['x_entry'] <= X_CITY_MAX and Y_CITY_MIN <= row['y_entry'] <= Y_CITY_MAX:
        return 1
    else:
        return 0

#%% Read File
data = pd.read_csv('data_test.csv', index_col=0)

#%% Extract 15:00 not moving devices
data_not_moving = data[data['time_entry']==data['time_exit']]
data_not_moving['time_entry'] = pd.to_datetime(data_not_moving['time_entry'], format="%H:%M:%S").dt.time
data_not_moving['time_exit'] = pd.to_datetime(data_not_moving['time_exit'], format="%H:%M:%S").dt.time
data_not_moving = data_not_moving[data_not_moving['time_entry']>datetime.time(15, 0, 0)]
data_not_moving.to_csv('1500_notmoving.csv')

#%% Extract one row moving devices
data_one_row_moving = data.drop(data_not_moving.index)
freq = data_one_row_moving['hash'].value_counts()
freq[freq.values==1].keys().shape[0]
x = data_one_row_moving[data_one_row_moving['hash'].isin(np.array(freq[freq.values==1].keys()))]
x['time_entry'] = pd.to_datetime(x['time_entry'], format="%H:%M:%S").dt.time
x['time_exit'] = pd.to_datetime(x['time_exit'], format="%H:%M:%S").dt.time
data_oneRow_moving = x[x['time_exit']>datetime.time(15, 0, 0)]
data_oneRow_moving['city_center'] = data_oneRow_moving.apply (lambda row: check_exit(row), axis=1)
data_oneRow_moving['time_entry'] = pd.to_datetime(data_oneRow_moving['time_entry'], format="%H:%M:%S").dt.time
data_oneRow_moving['time_exit'] = pd.to_datetime(data_oneRow_moving['time_exit'], format="%H:%M:%S").dt.time
data_oneRow_moving['time_range_entry'] = data_oneRow_moving['time_entry'].apply(time_to_range)
data_oneRow_moving['time_range_exit'] = data_oneRow_moving['time_exit'].apply(time_to_range)
data_oneRow_moving['x_entry']=normalize_x(data_oneRow_moving[['x_entry']].values)
data_oneRow_moving['y_entry']=normalize_y(data_oneRow_moving[['y_entry']].values)
data_oneRow_moving = data_oneRow_moving.drop(['hash','trajectory_id', 'vmax', 'vmin','vmean', 'time_entry', 'time_exit', 'x_exit', 'y_exit'], axis=1)
data_oneRow_moving.to_csv('oneRow_1500_moving.csv')
trash = x[x['time_exit']<datetime.time(15, 0, 0)]

#%% rows with prev info
data_with_prev = data_one_row_moving.drop(data_oneRow_moving.index)
data_with_prev = data_with_prev.drop(trash.index)
data_with_prev = data_with_prev[~data_with_prev['hash'].isin(np.array(data_not_moving['hash']))]
data_with_prev_exit_point = data_with_prev.drop_duplicates(subset='hash', keep='last', inplace=False)
data_with_prev_exit_point['time_entry'] = pd.to_datetime(data_with_prev_exit_point['time_entry'], format="%H:%M:%S").dt.time
data_with_prev_exit_point['time_exit'] = pd.to_datetime(data_with_prev_exit_point['time_exit'], format="%H:%M:%S").dt.time
data_with_prev_exit_point['time_range_entry'] = data_with_prev_exit_point['time_entry'].apply(time_to_range)
data_with_prev_exit_point['time_range_exit'] = data_with_prev_exit_point['time_exit'].apply(time_to_range)
data_with_prev_exit_point['time_range_entry'] = normalize_time(data_with_prev_exit_point['time_range_entry'])
data_with_prev_exit_point['time_range_exit'] = normalize_time(data_with_prev_exit_point['time_range_exit'])
data_with_prev = data_with_prev.drop(data_with_prev_exit_point.index)
data_with_prev = data_with_prev.drop_duplicates(subset='hash', keep='last', inplace=False)

# Data Preprocessing
data_with_prev['time_entry'] = pd.to_datetime(data_with_prev['time_entry'], format="%H:%M:%S").dt.time
data_with_prev['time_exit'] = pd.to_datetime(data_with_prev['time_exit'], format="%H:%M:%S").dt.time
data_with_prev['time_range_entry'] = data_with_prev['time_entry'].apply(time_to_range)
data_with_prev['time_range_exit'] = data_with_prev['time_exit'].apply(time_to_range)
data_with_prev['time_range_entry'] = normalize_time(data_with_prev['time_range_entry'])
data_with_prev['time_range_exit'] = normalize_time(data_with_prev['time_range_exit'])
data_with_prev['city_center_entry'] = data_with_prev.apply (lambda row: check_entry(row), axis=1)
data_with_prev['city_center'] = data_with_prev.apply (lambda row: check_exit(row), axis=1)
data_with_prev['x_entry']=normalize_x(data_with_prev[['x_entry']].values)
data_with_prev['y_entry']=normalize_y(data_with_prev[['y_entry']].values)
data_with_prev['x_exit']=normalize_x(data_with_prev[['x_exit']].values)
data_with_prev['y_exit']=normalize_y(data_with_prev[['y_exit']].values)
data_with_prev['exitPoint_x']=normalize_x(data_with_prev_exit_point[['x_entry']].values)
data_with_prev['exitPoint_y']=normalize_y(data_with_prev_exit_point[['y_entry']].values)
data_with_prev['exitPoint_time_range_entry'] = data_with_prev_exit_point[['time_range_entry']].values
data_with_prev['exitPoint_time_range_exit'] = data_with_prev_exit_point[['time_range_exit']].values
data_with_prev = data_with_prev.drop(['hash','trajectory_id', 'vmax', 'vmin',
    'vmean', 'time_entry', 'time_exit'], axis=1)
data_with_prev.to_csv('test_withPrevPos.csv')
data_with_prev_exit_point.to_csv('test_withPrevPos_exitPoint.csv')

#%%
data_with_prev.min()
