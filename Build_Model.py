from keras import layers, models, regularizers
import keras
from keras.utils import to_categorical,  plot_model
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import datetime
import pydot

#%% City range
X_CITY_MIN =
 3750901.5068
X_CITY_MAX = 3770901.5068
Y_CITY_MIN = -19268905.6133
Y_CITY_MAX = -19208905.6133

#%% Functions
def check_entry(row):
    if X_CITY_MIN <= row['x_entry'] <= X_CITY_MAX and Y_CITY_MIN <= row['y_entry'] <= Y_CITY_MAX:
        return 1
    else:
        return 0

def showModelSummary(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    # Plotting Training and Validation loss
    plt.plot(epochs, loss, 'ko', label="Training loss")
    plt.plot(epochs, val_loss, 'k', label="Validation loss")
    plt.title("Training and Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plotting Trainin and Validation accuracy
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.plot(epochs, acc, 'ko', label='Training accuracy')
    plt.plot(epochs, val_acc, 'k-', label='Validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

#%% Get data
data = pd.read_csv('data_train.csv')
train = pd.read_csv('prev_position_with_timeRangeAndCityCentered.csv', index_col=0)
target = pd.read_csv('last_position_with_timeRangeAndCityCentered.csv', index_col=0)

#%% Preprocessing
x_train = np.array(train)
train.head()
y_train = np.array(target['city_center'])
y_binary = to_categorical(y_train)
x_train_reshape = x_train.reshape((x_train.shape[0], 1, x_train.shape[1], 1))
SLICE = int(round(x_train.shape[0]*0.9))
x_val = x_train[SLICE:]
partial_x_train = x_train[:SLICE]
y_val = y_binary[SLICE:]
partial_y_train = y_binary[:SLICE]
x_val_reshape = x_train_reshape[SLICE:]
partial_x_train_reshape = x_train_reshape[:SLICE]
train.head()

#%% Covolutional Network Attempt
cnn = models.Sequential()
cnn.add(layers.Conv2D(64, (2, 1),
    padding="same",
    activation="relu",
    input_shape=(1, 12, 1)))
cnn.add(layers.Conv2D(64, (2, 1), padding="same", activation="relu"))
cnn.add(layers.MaxPooling2D(pool_size=(2,1),  padding='same', data_format='channels_last'))
cnn.add(layers.Conv2D(128, (2, 1), padding="same", activation="relu"))
cnn.add(layers.MaxPooling2D(pool_size=(2,1),  padding='same', data_format='channels_last'))
cnn.add(layers.Conv2D(256, (2, 1), padding="same", activation="relu"))
cnn.add(layers.MaxPooling2D(pool_size=(2,1),  padding='same', data_format='channels_last'))
cnn.add(layers.Flatten())
cnn.add(layers.Dense(512, activation="relu"))
cnn.add(layers.Dropout(0.2))
cnn.add(layers.Dense(256, activation="relu"))
cnn.add(layers.Dropout(0.2))
cnn.add(layers.Dense(128, activation="relu"))
cnn.add(layers.Dropout(0.2))
cnn.add(layers.Dense(2, activation="sigmoid"))

#%% Complie and fit
es = EarlyStopping(monitor='val_loss', verbose=1, mode='min', patience=5)
cnn.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=0.0005), metrics=['accuracy'])
history_cnn = cnn.fit(partial_x_train_reshape, partial_y_train,
    epochs=13, batch_size=512, shuffle=True, validation_data=(x_val_reshape, y_val), callbacks=[es])

#%% Plotting Result for CNN network
showModelSummary(history_cnn)
results_cnn = cnn.evaluate(x_val_reshape, y_val)
print (results_cnn)

#%% Predict data w/ prev pos
test = pd.read_csv('test_withPrevPos.csv', index_col=0)
test.head()
test_array = np.array(test)
test_array = test_array.reshape((test.shape[0], 1, 12, 1))
test_result = cnn.predict(test_array)

#%% Ready output csv
data_test = pd.read_csv('data_test.csv', index_col=0)
exitPoint = pd.read_csv('test_withPrevPos_exitPoint.csv', index_col=0)
trajectory_list = data_test.iloc[np.array(exitPoint.index),1].values
test_result_int = np.argmax(test_result, axis=1)

#%% Start output size:33515
output = pd.DataFrame({ 'id': trajectory_list, 'target': test_result_int})
output.shape[0]
# Add not moving
output2 = pd.read_csv('non_mov_1500_wtarget.csv')
output2.shape[0]
output3 = pd.read_csv('oneRow_moving_result.csv', index_col=0)
output3.shape[0]
# Concantenate
final = pd.concat([output, output2, output3])
final.shape[0]
final.to_csv('Attempt'+str(int(round(results_cnn[1]*10000)))+'.csv')
final['target'].value_counts()
