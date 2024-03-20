import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

num_features = 5

X, y = make_classification(100, num_features, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40,stratify=y)

model = Sequential()
model.add(Dense(60, input_shape=(num_features,), activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)
