from keras import Sequential
from keras.layers import Dense
from sklearn.datasets import make_classification, make_regression
import numpy as np
from sklearn.model_selection import train_test_split

# X, y = make_classification(1000, 4)

X, y = make_regression(1000, 4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=150, batch_size=10)
# evaluate the keras model
# _, accuracy = model.evaluate(X, y)
# print('Accuracy: %.2f' % (accuracy * 100))

# make class predictions with the model
# predictions = model.predict(X)
