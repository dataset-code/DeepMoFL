from keras.models import Sequential
from keras.layers import Dense
import numpy as np

np.random.seed(7)
T = 1000
X = np.array(range(T))
Y = np.sin(3.5 * np.pi * X / T) 

input_dim = 1

model = Sequential()
model.add(Dense(10, input_dim = input_dim, activation='tanh'))
model.add(Dense(90, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(1, activation='tanh'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=50, batch_size=10)