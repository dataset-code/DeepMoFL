from keras.layers import Dense, LSTM
from keras import Sequential
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_regression(100, 40000*7, n_targets=2, random_state=42)
X = X.reshape(100, 40000, 7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(40000, 7), dropout=0.2))
model.add(LSTM(128, return_sequences= True, dropout=0.2))
model.add(LSTM(128, return_sequences= False, dropout=0.2))
model.add(Dense(25))
model.add(Dense(2, activation='linear'))

model.compile(optimizer='adam', loss='mean_absolute_error')
model.summary()

model.fit(X_train, y_train, shuffle=False, verbose=1, epochs=5)

# prediction = model.predict(X_test, verbose=0)
# print(prediction)
