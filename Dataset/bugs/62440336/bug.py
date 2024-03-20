import tensorflow as tf
import keras
from sklearn.datasets import make_regression
import numpy as np
from sklearn.model_selection import train_test_split
 
X, y = make_regression(100, 1323000, random_state=42)
X = X.reshape(-1, 1323000, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = keras.Sequential([
    keras.layers.Conv1D(filters=100, kernel_size=10000, strides=5000, input_shape=(1323000, 1), activation='relu'),
    keras.layers.Conv1D(filters=100, kernel_size=10, strides=3, input_shape=(263, 100), activation='relu'),
    keras.layers.LSTM(1000),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dense(250, activation='relu'),
    keras.layers.Dense(1, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1000)
