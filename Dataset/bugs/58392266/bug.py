from sklearn.datasets import make_classification
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(100, 57 * 57 * 3, random_state=42)
X = X.reshape(-1, 57, 57, 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

vgg16_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(57, 57, 3)))
vgg16_model.summary()

model = Sequential()

for layer in vgg16_model.layers:
    layer.trainable = False
    model.add(layer)

model.add(Flatten())

model.add(Dense(4096, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=1e-5),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, validation_split=0.1, verbose=2)
