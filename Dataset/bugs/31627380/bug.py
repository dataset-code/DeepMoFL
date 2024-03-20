from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
import numpy as np 
from sklearn import preprocessing

### load train and test ###
train  = pd.read_csv('train.csv', index_col=0)
test  = pd.read_csv('test.csv', index_col=0)

y_train = train.Survived
train.drop('Survived', axis=1, inplace=True)

columns = train.columns
test_ind = test.index

train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())
train['Fare'] = train['Fare'].fillna(train['Fare'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

category_index = [0,1,2,4,5,6,8,9]
for i in category_index:
    # print str(i)+" : "+columns[i]
    train[columns[i]] = train[columns[i]].fillna('missing')
    test[columns[i]] = test[columns[i]].fillna('missing')

train = np.array(train)
test = np.array(test)

### label encode the categorical variables ###
for i in category_index:
    # print str(i)+" : "+str(columns[i])
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[:,i]) + list(test[:,i]))
    train[:,i] = lbl.transform(train[:,i])
    test[:,i] = lbl.transform(test[:,i])

### making data as numpy float ###
train = train.astype(np.float32)
test = test.astype(np.float32)

model = Sequential()
model.add(Dense(len(columns), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam")
model.fit(train, y_train, epochs=10, batch_size=4, validation_split=0.20)
