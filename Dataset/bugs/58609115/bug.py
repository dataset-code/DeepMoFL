from keras import Sequential
from keras.layers import Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(1000, 1, random_state=42, n_informative=1,n_redundant=0,n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=33)

classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(3557, activation='sigmoid', kernel_initializer='random_normal', input_dim=1))
#Second  Hidden Layer
classifier.add(Dense(3557, activation='sigmoid', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(2, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Fitting the data to the training dataset
classifier.fit(X_train,y_train, validation_split=0.1,batch_size=50, epochs=10)
