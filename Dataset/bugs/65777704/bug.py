from tensorflow.keras import Sequential
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical, normalize
import numpy as np
 
(x_train, y_train), (x_test, y_test) = load_data()
x_train = normalize(x_train)
x_test = normalize(x_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Flatten())
model.add(Dense(units=8, activation='sigmoid'))
model.add(Dense(units=16, activation='sigmoid'))
model.add(Dense(units=32, activation='sigmoid'))
model.add(Dense(units=64, activation='sigmoid'))
model.add(Dense(units=128, activation='sigmoid'))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000, batch_size=500)
