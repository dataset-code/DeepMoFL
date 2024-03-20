from keras import Input, regularizers, Model
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import train_test_split

input_shape = (128, 128, 3)
X, y = make_classification(500, 128 * 128 * 3,random_state=42)
X = X.reshape(-1, 128, 128, 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

inp = Input(shape=input_shape)
out = Conv2D(16, (5, 5), activation='relu', kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(0.01), padding='same')(inp)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.5)(out)

out = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(0.01), padding='same')(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.5)(out)

out = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(0.01), padding='same')(out)
out = Dropout(0.5)(out)

out = Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(0.01), padding='same')(out)
out = Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(0.01), padding='same')(out)
out = MaxPooling2D(pool_size=(2, 2))(out)

out = Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(0.01), padding='same')(out)
out = Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(0.01), padding='same')(out)
out = MaxPooling2D(pool_size=(2, 2))(out)

out = Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(0.01), padding='same')(out)
out = Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(0.01), padding='same')(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Conv2D(512, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(0.01), padding='same')(out)
out = MaxPooling2D(pool_size=(2, 2))(out)

out = Flatten()(out)
out = Dropout(0.5)(out)
dense1 = Dense(1, activation="softmax")(out)
model = Model(inputs=inp, outputs=dense1)

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['acc'])
model.fit(X_train, y_train, epochs=10, validation_split=0.1)
