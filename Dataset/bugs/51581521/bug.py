from keras import Input, applications, Model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import os

img_width, img_height = 150, 150
nb_train_samples = 200

x, y = make_classification(nb_train_samples, img_width * img_height * 3, random_state=42)
x = x.reshape(-1, 150, 150, 3)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

input_tensor = Input(shape=(150, 150, 3))

top_model = Sequential()

base_model = applications.VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
for layer in base_model.layers:
    layer.trainable = False
    top_model.add(layer)

top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation="relu"))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='softmax'))
# model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

top_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
top_model.fit(X_train, y_train, epochs=50, batch_size=64)
