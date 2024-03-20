import numpy as np
from keras.layers import Input, Dense
from keras.models import Sequential

np.random.seed(7)

dataSize = 1000
variables = 2
data = np.zeros((dataSize,variables))
data[:, 0] = np.random.uniform(0, 0.8, size=dataSize)
data[:, 1] = np.random.uniform(0, 0.1, size=dataSize)
trainData, testData = data[:900], data[900:]

model = Sequential()
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['mse'])

model.fit(trainData, trainData, epochs=20)
