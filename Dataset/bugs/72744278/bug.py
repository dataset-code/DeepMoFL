from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(1000, 50*50*3, random_state=42)
X =X.reshape(1000,50,50,3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = Sequential()
#model.add(Input(shape=(50,50,3)))
# for i in range(hp.Int('num_blocks', 1, 2)):
#     hp_padding=hp.Choice('padding_'+ str(i), values=['valid', 'same'])
#     hp_filters=hp.Choice('filters_'+ str(i), values=[32, 64])

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(50, 50, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())

# hp_units = hp.Int('units', min_value=25, max_value=150, step=25)
model.add(Dense(25, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10,activation="softmax"))

model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])

model.fit(X_train,y_train,validation_split=0.1,epochs=10)
