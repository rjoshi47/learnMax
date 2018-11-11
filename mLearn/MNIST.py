import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import tensorflow as tf
np.random.seed(1)

def displayNum(displaySetX, displaySetY, index):
    a = np.reshape(displaySetX[index], (28,28))
    print ("y = " + str(displaySetY[index]))
    plt.imshow(a)
    plt.show()

def load_dataset(testSize):
    full_dataset = pd.read_csv('E:\\MachineMax\\MNIST\\all\\train.csv').sample(frac=1) # read and shuffle the data
    trainSet =  full_dataset[:-int(testSize*len(full_dataset))]
    testSet =  full_dataset[-int(testSize*len(full_dataset)):]

    train_set_x_orig = np.array(trainSet.drop(['label'], 1))# your train set features
    train_set_y_orig = np.array(trainSet["label"][:]) # your train set labels

    test_set_x_orig = np.array(testSet.drop(['label'], 1))# your train set features
    test_set_y_orig = np.array(testSet["label"][:]) # your train set labels

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig

def normalize(X_train_orig, test_set_x_orig):
    X_train = X_train_orig/np.max(X_train_orig)
    X_test = test_set_x_orig/np.max(test_set_x_orig)

    X_train -= np.std(X_train)
    X_test -= np.std(X_test)

    return X_train, X_test

train_set_x_orig, train_set_y_orig, test_set_x_orig, Y_test = load_dataset(0.2)
X_train, X_test = normalize(train_set_x_orig, test_set_x_orig)

Y_train = np_utils.to_categorical(train_set_y_orig)
#Y_test = np_utils.to_categorical(Y_test)

model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(192))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.fit(X_train, Y_train, epochs=10, batch_size=16, verbose=2)

preds = model.predict_classes(X_test, verbose=1)

c = 0
for i in range(len(preds)):
    if preds[i] == Y_test[i]:
        c += 1
print(c/len(preds))
