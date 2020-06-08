#%% import package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% read data
data = pd.read_csv('data_train.csv', index_col=0)

#%% manipulaing dataframe
data_hashList = data['hash'].unique()
data_hashList
for j in range(50):
    x_in = np.array(data[data['hash']==data_hashList[j]]['x_entry'])
    y_in = np.array(data[data['hash']==data_hashList[j]]['y_entry'])
    x_out = np.array(data[data['hash']==data_hashList[j]]['x_exit'])
    y_out = np.array(data[data['hash']==data_hashList[j]]['y_exit'])
    for i in range(0, len(x_in)):
        x1, x2 = x_in[i], x_out[i]
        y1, y2 = y_in[i], y_out[i]
        plt.plot([x1,x2],[y1,y2],'k-')
    plt.show()
