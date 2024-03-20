from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import train_test_split
 
X, y = make_classification(1000, 64 * 64, n_classes=4, n_informative=3,random_state=42)
X = X.reshape(-1, 64, 64, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(64, 64, 1), activation='relu'))
# model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3),activation='relu'))
# model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3),activation='relu'))
# model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64,activation='relu'))
# model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))
# model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)
