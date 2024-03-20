from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.optimizers import SGD
import numpy as np

MAX_NB_WORDS = 10
embedding_vecor_length = 32
max_length = 10
batch_size = 1
input_dim = max_length

X_train = [[0, 1, 1, 1, 1, 0, 0, 0, 0, 0,],
[0, 1, 1, 1, 1, 0, 0, 0, 0, 0,], 
[0, 1, 1, 1, 0, 0, 0, 0, 0, 0,],
[0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
[0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
[0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]]

y_train = [[1],[1],[1],[0],[0],[1],[1]]
X_train = np.array(X_train)
y_train = np.array(y_train)



model = Sequential()
model.add(Embedding(MAX_NB_WORDS, embedding_vecor_length, input_length=max_length,batch_input_shape=( batch_size, input_dim)))
model.add(LSTM(10, return_sequences=False))
model.add(Dense(1, activation='softmax'))

opt = SGD(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=60, batch_size=batch_size)

