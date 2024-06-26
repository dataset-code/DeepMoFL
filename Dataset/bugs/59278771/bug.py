import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import numpy as np

seed = 7
numpy.random.seed(seed)

from sklearn.datasets import load_iris

X, encoded_Y = load_iris(return_X_y=True)
mms = MinMaxScaler()
X = mms.fit_transform(X)

dummy_y = np_utils.to_categorical(encoded_Y)


def baseline_model():
    
    model = Sequential()
    model.add(Dense(4, input_dim=4, activation="relu", kernel_initializer="normal"))
    model.add(Dense(3, activation="sigmoid", kernel_initializer="normal"))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=200, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print(results)


model = baseline_model()
model.fit(X, dummy_y)

