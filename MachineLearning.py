import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from time import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from operator import itemgetter

#%% Import Data
train = pd.read_csv('non_normalize_prev_position_with_timeRangeAndCityCentered.csv', index_col=0)
target = pd.read_csv('non_normalize_last_position_with_timeRangeAndCityCentered.csv', index_col=0)

#%% Slicing
X = np.array(train)
y = np.array(target['city_center'])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
train.head()
#%%
seed = np.random.seed(1234)

#%% Random Forest
clf = RandomForestClassifier(random_state=seed, max_depth=10,
    n_estimators=300, min_samples_split=2, min_samples_leaf=1, n_jobs=1, criterion='gini')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

#%% Test Set
test = pd.read_csv('non_normalize_test_withPrevPos.csv', index_col=0)
test_array = np.array(test)
test_result = clf.predict(test_array)

#%%
f = pd.read_csv("Attempt RF.csv", index_col=0)
f = f['target']
f.value_counts()

#%% Ready output csv
data_test = pd.read_csv('data_test.csv', index_col=0)
exitPoint = pd.read_csv('test_withPrevPos_exitPoint.csv', index_col=0)
trajectory_list = data_test.iloc[np.array(exitPoint.index),1].values

#%% Start output size:33515
output = pd.DataFrame({ 'id': trajectory_list, 'target': test_result})
output.shape[0]
# Add not moving
output2 = pd.read_csv('non_mov_1500_wtarget.csv')
output2.shape[0]
output3 = pd.read_csv('oneRow_moving_result.csv', index_col=0)
output3.shape[0]
# Concantenate
final = pd.concat([output, output2, output3])
final.shape[0]
final.to_csv('Attempt RF 05082019'+'.csv')
final['target'].value_counts()
