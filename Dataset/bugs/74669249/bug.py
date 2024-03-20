from keras import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, BatchNormalization
import keras
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_regression(4413, 71*19, n_targets=2, random_state=42)
X = X.reshape(4413, 71, 19)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(71,19)))
model.add(Dropout(.2))
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True, input_shape=(71,19)))
model.add(Dropout(.2))
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=False, input_shape=(71,19)))
model.add(Dropout(.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(.2))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# print(model.summary())
model.fit(X_train, y_train, epochs=5)