import keras
import numpy as np
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
train_images = np.zeros((*x_train.shape, 3)) # orig shape: (60 000, 28, 28, 1) -> rgb shape: (60 000, 28, 28, 3)
for num in range(x_train.shape[0]):
    rgb = np.random.randint(3) 
    train_images[num, ..., rgb] = x_train[num]/255
test_images = np.zeros((*x_test.shape, 3))
for num in range(x_test.shape[0]):
    rgb = np.random.randint(3) 
    test_images[num, ..., rgb] = x_test[num]/255

model = keras.Sequential()
model.add(keras.layers.Conv2D(64, kernel_size=(3,3), padding='same'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Conv2D(64, kernel_size=(3,3), padding='same'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2), padding='same'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Softmax())

input_shape = train_images.shape
model.build(input_shape)

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(train_images, y_train, batch_size=12, epochs=25)
