from keras import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
import numpy as np
from sklearn.model_selection import train_test_split

X, y = make_regression(1000, 4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = Sequential()
model.add(Dense(30, activation='relu', input_shape=(4,)))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()

print('Compile & fit')
model.compile(loss='mean_squared_error', optimizer='RMSprop')
model.fit(X_train, y_train, batch_size=128, epochs=13)
