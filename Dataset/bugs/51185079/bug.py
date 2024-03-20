from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np 

x, y = make_classification(1000, 34, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2)

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=34))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=2, batch_size=200)
