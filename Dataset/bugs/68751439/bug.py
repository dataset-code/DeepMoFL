import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import train_test_split

X_train, y_train = make_classification(100, 14,random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# define the keras model
model1 = keras.Sequential()
model1.add(keras.layers.Dense(64, input_dim=14, activation='relu'))
model1.add(keras.layers.Dense(128, activation='relu'))
model1.add(keras.layers.Dense(64, activation='relu'))  
model1.add(keras.layers.Dense(1, activation='softmax'))

# compile the keras model
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
performance1 = model1.fit(X_train, y_train, epochs=100, validation_split=0.2)
