from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(1000, 100*100*3, random_state=42, n_classes=3, n_informative=100)
X =X.reshape(1000,100,100,3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4,stratify=y)

model = Sequential()
model.add(Conv2D(128, 3, activation="relu", input_shape=(100,100,3)))
model.add(MaxPooling2D())
model.add(Conv2D(64, 3, activation="relu"))
model.add(Conv2D(32, 3, activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.50))
model.add(Flatten())
model.add(Dense(5000, activation = "relu"))
model.add(Dense(1000, activation = "relu"))
model.add(Dense(131, activation = "softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=10, validation_split=0.1)

