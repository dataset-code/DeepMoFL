from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, InputLayer, BatchNormalization
import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from keras.optimizers import Adam
from keras.utils import to_categorical

X, y = make_classification(1000, 28*28, random_state=42, n_classes=10,n_informative=100)
X =X.reshape(1000,28,28,1)
y=to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# Define the network
model = Sequential()
model.add(InputLayer(input_shape=(28,28,1)))
# model.add(Augmentations1(p=0.5, freq_type='mel', max_aug=2))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())

model.add(Flatten())
# model.add(Dense(numClasses, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='sigmoid'))

model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer=Adam(learning_rate=0.001),
    run_eagerly=False)  # this parameter allows to debug and use regular functions inside layers: print(), save() etc..

model.fit(X_train, y_train, validation_split=0.1,epochs=10)
