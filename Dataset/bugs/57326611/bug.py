import tensorflow as tf
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(10000, 5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#preprocessing_layer = tf.keras.layers.DenseFeatures(feature_columns)
preprocessing_layer = tf.keras.layers.InputLayer(input_shape=(5,))

model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20)