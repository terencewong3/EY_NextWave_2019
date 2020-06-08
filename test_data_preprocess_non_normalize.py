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
data_not_moving = data[data['time_entry']==data['time_exit']].copy()
data_not_moving['time_entry'] = pd.to_datetime(data_not_moving['time_entry'], format="%H:%M:%S").dt.time
data_not_moving['time_exit'] = pd.to_datetime(data_not_moving['time_exit'], format="%H:%M:%S").dt.time
data_not_moving = data_not_moving[data_not_moving['time_entry']>datetime.time(15, 0, 0)]
data_not_moving.to_csv('1500_notmoving.csv')

#%% Extract one row moving devices
data_one_row_moving = data.drop(data_not_moving.index)
freq = data_one_row_moving['hash'].value_counts()
freq[freq.values==1].keys().shape[0]
x = data_one_row_moving[data_one_row_moving['hash'].isin(np.array(freq[freq.values==1].keys()))].copy()
x['time_entry'] = pd.to_datetime(x['time_entry'], format="%H:%M:%S").dt.time
x['time_exit'] = pd.to_datetime(x['time_exit'], format="%H:%M:%S").dt.time
data_oneRow_moving = x[x['time_exit']>datetime.time(15, 0, 0)]
data_oneRow_moving['city_center'] = data_oneRow_moving.apply(lambda row: check_exit(row), axis=1)
data_oneRow_moving['time_entry'] = pd.to_datetime(data_oneRow_moving['time_entry'], format="%H:%M:%S").dt.time
data_oneRow_moving['time_exit'] = pd.to_datetime(data_oneRow_moving['time_exit'], format="%H:%M:%S").dt.time
#data_oneRow_moving['time_range_entry'] = data_oneRow_moving['time_entry'].apply(time_to_range)
#data_oneRow_moving['time_range_exit'] = data_oneRow_moving['time_exit'].apply(time_to_range)
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
data_with_prev_exit_point['time_entry'] = pd.to_timedelta(data_with_prev_exit_point['time_entry']).dt.total_seconds()
data_with_prev_exit_point['time_exit'] = pd.to_timedelta(data_with_prev_exit_point['time_exit']).dt.total_seconds()
data_with_prev = data_with_prev.drop(data_with_prev_exit_point.index)
data_with_prev = data_with_prev.drop_duplicates(subset='hash', keep='last', inplace=False)

# Data Preprocessing
data_with_prev['time_entry'] = pd.to_timedelta(data_with_prev['time_entry']).dt.total_seconds()
data_with_prev['time_exit'] = pd.to_timedelta(data_with_prev['time_exit']).dt.total_seconds()
data_with_prev['city_center_entry'] = data_with_prev.apply (lambda row: check_entry(row), axis=1)
data_with_prev['city_center'] = data_with_prev.apply (lambda row: check_exit(row), axis=1)
data_with_prev['exitPoint_x']=data_with_prev_exit_point[['x_entry']].values
data_with_prev['exitPoint_y']=data_with_prev_exit_point[['y_entry']].values
data_with_prev['exitPoint_time_entry'] = data_with_prev_exit_point[['time_entry']].values
data_with_prev['exitPoint_time_exit'] = data_with_prev_exit_point[['time_exit']].values
data_with_prev = data_with_prev.drop(['hash','trajectory_id', 'vmax', 'vmin',
    'vmean'], axis=1)
data_with_prev.to_csv('non_normalize_test_withPrevPos.csv')
data_with_prev_exit_point.to_csv('non_normalize_test_withPrevPos_exitPoint.csv')


#%%
data_with_prev.head()
