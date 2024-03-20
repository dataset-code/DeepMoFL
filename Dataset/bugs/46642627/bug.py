from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy

# fix random seed for reproducibility
numpy.random.seed(7)

# split into input (X) and output (Y) variables
X = []
Y = []
count = 0

while count < 10000:
    count += 1
    X += [count / 10000]
    numpy.random.seed(count)
    #Y += [numpy.random.randint(1, 101) / 100]
    Y += [(count + 1) / 100]
print(str(X) + ' ' + str(Y))

# create model
model = Sequential()
model.add(Dense(50, input_dim=1, kernel_initializer = 'uniform', activation='relu'))
model.add(Dense(50, kernel_initializer = 'uniform', activation='relu'))
model.add(Dense(1, kernel_initializer = 'uniform', activation='sigmoid'))

# Compile model
opt = optimizers.SGD(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=100)
