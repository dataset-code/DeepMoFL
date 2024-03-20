from keras import Sequential, Model, Input
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.datasets import make_regression
import numpy as np
from sklearn.model_selection import train_test_split

m = 10
n = 3

X, y = make_regression(1000, n_features=m, n_targets=n, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.75, random_state=40)

inputs = Input(shape=(m,))
hidden = Dense(100, activation='sigmoid')(inputs)
hidden = Dense(80, activation='sigmoid')(hidden)
outputs = Dense(n, activation='softmax')(hidden)

opti = Adam(learning_rate=0.001)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=opti,
              loss='poisson',
              metrics=['accuracy'])

model.fit(X_train, y_train, verbose=2, batch_size=32, epochs=30)
