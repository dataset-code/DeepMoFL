from sklearn.datasets import make_classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
 
train_x, train_y = make_classification(1000, 10 * 150, random_state=42)
train_x = train_x.reshape(-1, 10, 150)
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=42, stratify=train_y)


model = Sequential()
model.add(Dense(32, input_shape=(len(train_x[0]), 150), activation="relu"))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(Flatten())
model.add(Dense(1, activation="softmax"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

hist = model.fit(X_train, y_train, epochs=200, batch_size=2, verbose=1)