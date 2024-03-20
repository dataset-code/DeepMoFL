from keras import Sequential
from keras.layers import Dense,Dropout
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import keras

X, y = make_classification(100, 10, random_state=42, n_classes=3, n_informative=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)

model = Sequential([
  Dense(128, activation='relu'),
  Dense(128, activation='relu'),
  Dropout(.1),
  Dense(1)
])

model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train,y_train,
          validation_data=(X_test, y_test),
          epochs=12)
