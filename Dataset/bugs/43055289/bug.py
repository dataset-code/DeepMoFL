import numpy as np
from keras import Sequential
from keras.layers import Conv2D, LeakyReLU, BatchNormalization, Flatten, Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


X, y = make_classification(60, 225 * 225 * 3, random_state=42)
X = X.reshape(-1, 225, 225, 3)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

# train_tensors, train_labels contain training data

model = Sequential()
model.add(Conv2D(filters=5,
                                  kernel_size=[4, 4],
                                  strides=2,
                                  padding='same',
                                  input_shape=[225, 225, 3]))
model.add(LeakyReLU(0.2))

model.add(Conv2D(filters=10,
                                  kernel_size=[4, 4],
                                  strides=2,
                                  padding='same'))
model.add(BatchNormalization(axis=3))
model.add(LeakyReLU(0.2))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=8, epochs=5, shuffle=True)


