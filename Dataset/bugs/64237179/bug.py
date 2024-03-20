from keras import Sequential
from keras.layers import Conv3D, BatchNormalization, LeakyReLU, MaxPooling3D, Dropout, Flatten, Dense, Activation
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
 
X, y = make_classification(420, 50 * 50 * 50, random_state=42)
X = X.reshape(-1, 50, 50, 50, 1)
shape = X.shape[1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


model = Sequential()
model.add(Conv3D(64, kernel_size=(5, 5, 5), activation='linear',
                                  kernel_initializer='glorot_uniform', input_shape=shape))
model.add(BatchNormalization(center=True, scale=True))
model.add(LeakyReLU(.1))
model.add(Dropout(.25))
model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='linear',
                                  kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(center=True, scale=True))
model.add(LeakyReLU(.1))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(.25))

model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='linear',
                                  kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(center=True, scale=True))
model.add(LeakyReLU(.1))
model.add(Dropout(.25))
model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='linear',
                                  kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(center=True, scale=True))
model.add(LeakyReLU(.1))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(.25))

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization(center=True, scale=True))
model.add(LeakyReLU(.1))
model.add(Dropout(.5))
model.add(Dense(512))
model.add(BatchNormalization(center=True, scale=True))
model.add(LeakyReLU(.1))
model.add(Dropout(.5))
model.add(Dense(1))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=50,verbose=1)

