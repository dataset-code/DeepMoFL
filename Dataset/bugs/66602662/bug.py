from keras.losses import CategoricalCrossentropy
import tensorflow as tf
from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import train_test_split
 
X, y = make_classification(1000, 5, n_classes=5, n_informative=4, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(8, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss=CategoricalCrossentropy(from_logits=True))
model.fit(X_train, y_train, epochs=20, batch_size=24)
