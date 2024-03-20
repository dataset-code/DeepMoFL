from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
import numpy as np
from sklearn.model_selection import train_test_split

X, y = make_regression(100, 18, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.75, random_state=42)

model = Sequential()
model.add(Dense(18, input_dim=18, activation='tanh'))
model.add(Dense(36, activation='relu'))
model.add(Dense(72, activation='relu'))
model.add(Dense(72, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='softmax'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model
model.fit(X_train, y_train, epochs=100, batch_size=35)
