import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('C:\\Users\\rjoshi\\Documents\\ml\\breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True) # this replace value might effect final result
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

test_data = np.array([[10,10,10,8,6,1,8,9,1], [3,1,1,1,2,1,2,1,1], [4,1,1,1,2,1,2,1,1]])
prediction = clf.predict(test_data.reshape(len(test_data), -1))
print(accuracy, prediction)