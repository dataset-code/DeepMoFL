from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import train_test_split


X, y = make_classification(1000, 283, n_classes=4, n_informative=4,random_state=42)
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.75, random_state=42)


model = Sequential()
model.add(Dense(100, input_dim=283, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(150, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, kernel_initializer='normal', activation='sigmoid'))
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5)

