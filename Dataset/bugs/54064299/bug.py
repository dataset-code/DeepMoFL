from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_regression(1000, 150 * 150 * 1, n_targets=4, random_state=42)
X = X.reshape(-1, 150, 150, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=(150, 150, 1),activation='relu'))
# model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(32, (3, 3),activation='relu'))
# model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Dense(64,activation='relu'))
# model.add(Activation("relu"))

model.add(Flatten())
model.add(Dense(4))

model.compile(loss="mean_squared_error", optimizer='adam', metrics=[])

model.fit(X_train, y_train, batch_size=1, validation_split=0, epochs=30, verbose=1)
