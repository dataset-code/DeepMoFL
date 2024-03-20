import keras
from keras.models import Sequential
from keras.layers import Reshape, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import tensorflow as tf
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(len(X_train),784)
X_test = X_test.reshape(len(X_test),784)
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)

model = Sequential()
bias_initializer = keras.initializers.Constant(value = 0.1)

neurons_nb_layer_1 = 32
neurons_nb_layer_2 = 64
neurons_nb_layer_3 = 1024

model.add(Reshape((28, 28, 1), input_shape=(784,)))
model.add(Conv2D(filters = neurons_nb_layer_1, kernel_size = 5*5, padding = 'same', activation = "relu", bias_initializer = bias_initializer))
model.add(MaxPooling2D(padding='same'))
model.add(Conv2D(filters = neurons_nb_layer_2, kernel_size = 5*5, padding = 'same', activation = "relu", bias_initializer = bias_initializer))
model.add(MaxPooling2D(padding='same'))
model.add(Reshape((1,7*7*neurons_nb_layer_2)))
model.add(Dense(units = neurons_nb_layer_3, activation = "relu", bias_initializer = bias_initializer))
model.add(Dropout(rate = 0.5))
model.add(Flatten())
model.add(Dense(units = 10, activation = "relu"))
 
model.summary()

model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = 'adam',
              metrics=['accuracy']
              )

model.fit(X_train,y_train,epochs=10)
