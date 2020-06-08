from keras import layers, models
import keras
from keras.utils import to_categorical
from keras.optimizers import SGD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

#%% City range
X_CITY_MIN = 3750901.5068
X_CITY_MAX = 3770901.5068
Y_CITY_MIN = -19268905.6133
Y_CITY_MAX = -19208905.6133

#%% Functions
def check_entry(row):
    if X_CITY_MIN <= row['x_entry'] <= X_CITY_MAX and Y_CITY_MIN <= row['y_entry'] <= Y_CITY_MAX:
        return 1
    else:
        return 0

#%% Get data
data = pd.read_csv('oneRow_training.csv', index_col=0)
train = data
target = data['city_center']
train.head()

#%% Preprocessing
x_train = np.array(train)
y_train = np.array(target)
y_binary = to_categorical(y_train)
target.value_counts()
#%% Start Mode
model = models.Sequential()
model.add(layers.Dense(46, activation='relu', input_shape=(5, )))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))

# Compile
model.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer = 'rmsprop')
# optimizer = keras.optimizers.Adam(lr=0.0001)
# Validating ApproachSLICE = 1110
x_val = x_train[SLICE:]
partial_x_train = x_train[:SLICE]
y_val = y_binary[SLICE:]
partial_y_train = y_binary[:SLICE]

# Training model
history = model.fit(partial_x_train, partial_y_train, epochs=80, batch_size=512,
                    shuffle=True, validation_data=(x_val, y_val))

model.summary()
#%% Plotting results
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

results = model.evaluate(x_val, y_val)
print (results)

#%% Predict data w/ prev pos
test = pd.read_csv('oneRow_1500_moving.csv', index_col=0)
test.head()
test_array = np.array(test)
test_result = model.predict(test_array)

#%% Ready output csv
data_test = pd.read_csv('data_test.csv', index_col=0)
trajectory_list = data_test.iloc[np.array(test.index),1].values
test_result_int = np.argmax(test_result, axis=1)

#%% Output
output = pd.DataFrame({ 'id': trajectory_list, 'target': test_result_int})
output.to_csv('oneRow_moving_result.csv')
output['target'].value_counts()
