from keras.layers import Dense, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np


X, y = make_classification(1000, 30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=40,stratify=y)

model = Sequential()
model.add(Dense(600, input_shape=(30,),activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
# model.add(Dense(25, input_shape=(30,),activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=100)
