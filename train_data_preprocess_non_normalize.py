import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import datetime
from sklearn import preprocessing

data_raw = pd.read_csv('data_train.csv', index_col=0)

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
# def time_to_numerical(x):

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

def normalize_time(t):
    return (t-1)/(32-1)

#%% Find one row only moving devices
# Basically all of the data is at ard 15:00
freq = data_raw['hash'].value_counts()
data_one_row = data_raw[data_raw['hash'].isin(np.array(freq[freq.values==1].keys()))].copy()
data_one_row['city_center'] = data_one_row.apply (lambda row: check_exit(row), axis=1)
data_one_row['time_entry'] = pd.to_timedelta(data_one_row['time_entry']).dt.total_seconds()
data_one_row['time_exit'] = pd.to_timedelta(data_one_row['time_exit']).dt.total_seconds()
data_one_row.shape[0]

#%% Extract Last and prev position for each devices:
data_with_prev = data_raw[~data_raw['hash'].isin(np.array(data_one_row['hash']))].copy()
data_last_position = data_with_prev.drop_duplicates(subset='hash', keep='last', inplace=False).copy()
data_prev_position = data_with_prev.drop(data_last_position.index)
data_prev_position.drop_duplicates(subset='hash', keep='last', inplace=True)
data_last_position['time_entry'] = pd.to_timedelta(data_last_position['time_entry']).dt.total_seconds()
data_last_position['time_exit'] = pd.to_timedelta(data_last_position['time_exit']).dt.total_seconds()
data_last_position['city_center'] = data_last_position.apply (lambda row: check_exit(row), axis=1)

data_prev_position['time_entry'] = pd.to_timedelta(data_prev_position['time_entry']).dt.total_seconds()
data_prev_position['time_exit'] = pd.to_timedelta(data_prev_position['time_exit']).dt.total_seconds()
data_prev_position['city_center_entry'] = data_prev_position.apply (lambda row: check_entry(row), axis=1)
data_prev_position['city_center'] = data_prev_position.apply (lambda row: check_exit(row), axis=1)
data_prev_position['exitPoint_x'] = data_last_position[['x_entry']].values
data_prev_position['exitPoint_y'] = data_last_position[['y_entry']].values
data_prev_position['exitPoint_time_entry'] = np.array(data_last_position[['time_entry']])
data_prev_position['exitPoint_time_exit'] = np.array(data_last_position[['time_exit']])
data_prev_position = data_prev_position.drop(['hash','trajectory_id', 'vmax', 'vmin','vmean'], axis=1)
data_prev_position.head()

#%% output
data_one_row = data_one_row.drop(['hash','trajectory_id', 'vmax', 'vmin','vmean', 'time_entry', 'time_exit', 'x_exit', 'y_exit'], axis=1)
# data_one_row.to_csv('oneRow_training.csv')
data_last_position.to_csv('non_normalize_last_position_with_timeRangeAndCityCentered.csv')
data_prev_position.to_csv('non_normalize_prev_position_with_timeRangeAndCityCentered.csv')
data_with_prev.to_csv('data_train_wo_oneRow.csv')

data_prev_position.min()
