from keras import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPool3D
from keras.optimizers import Nadam
from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import train_test_split
 
X, y = make_classification(100, 100 * 100 * 3 * 6, random_state=42)
X = X.reshape(-1, 100, 100, 3, 6)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

model = Sequential()
model.add(Conv3D(32, (3, 3, 3), input_shape=(100, 100, 3, 6), activation='linear', padding='same'))
model.add(MaxPool3D((2, 2, 2), padding='same'))
model.add(Conv3D(32, (3, 3, 3), activation='linear', padding='same'))
model.add(MaxPool3D((2, 2, 2), padding='same'))
model.add(Conv3D(32, (3, 3, 3), activation='linear', padding='same'))
model.add(Flatten())
model.add(Dense(10, activation='linear'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer=Nadam(), loss='mape')
model.fit(X_train, y_train, epochs=10, shuffle=True, batch_size=20)