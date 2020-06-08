#%% Start Mode
dropout_rate = 0.3
model = models.Sequential()
model.add(layers.Dense(46, activation='relu', input_shape=(10,)))
model.add(layers.Dropout(dropout_rate))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(dropout_rate))
model.add(layers.Dense(24, activation='relu'))
model.add(layers.Dropout(dropout_rate))
model.add(layers.Dense(2, activation='sigmoid'))

# Compile
model.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer = keras.optimizers.Adam(lr=0.001))
# Training model
history = model.fit(partial_x_train, partial_y_train, epochs=50, batch_size=1024,
                    shuffle=True, validation_data=(x_val, y_val))

#%% Plotting results
showModelSummary(history)
results = model.evaluate(x_val, y_val)
print (results)
