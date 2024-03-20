from keras import Sequential
from keras.layers import Dense,Dropout
import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(1000, 100*100*3, random_state=42)
X = X.reshape(1000,100,100,3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=30)

obj_classes = 9

model = keras.Sequential()

# building the convolution layers
model.add(keras.layers.Conv2D(32,(3,3),input_shape= (100,100,3),padding='same',activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Conv2D(64,(3,3), padding='same',activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Conv2D(128,(3,3), padding='same',activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Conv2D(256,(3,3), padding='same',activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Conv2D(512,(3,3), padding='same',activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Flatten())
    
# building the dense layers
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(keras.layers.Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(keras.layers.Dense(256, activation='relu'))
# model.add(Dropout(0.6))
model.add(keras.layers.Dense(128, activation='relu'))
# model.add(Dropout(0.6))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(obj_classes,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=64,epochs=50, verbose=1)