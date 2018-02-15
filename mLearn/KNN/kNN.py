from numpy import *
import operator
import matplotlib.pyplot as plt

def createDataSet():
    group = array([[0, 0], [10, 0], [10, 10], [0, 10], [5,5]])
    labels = ['A', 'B', 'C', 'D', 'E']
    return group, labels

def classify(node, dataSet, labels, k):
    dataSize = dataSet.shape[0]
    diffMat = tile(node, (dataSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    distances = sqDiffMat.sum(axis=1) # array of distances
    sortedDistances = distances.argsort() # sorted on indexes

    labelCountDict = {}
    for i in range(k):
        cLabel = labels[sortedDistances[i]]
        labelCountDict[cLabel] = labelCountDict.get(cLabel, 0) + 1

    sortedCount = sorted(labelCountDict.items(),
                         key=operator.itemgetter(1), reverse=True)
    #print(sortedCount)
    return sortedCount[0][0]

colors = {}
colors['A'] = 'r'
colors['B'] = 'b'
colors['C'] = 'y'
colors['D'] = 'g'
colors['E'] = 'm'

(groups, labels) = createDataSet()

x = arange(0, 10, 0.3)
y = arange(0, 10, 0.3)
for i in range(len(x)):
    for j in range(len(y)):
        cLabel = classify([x[i],y[j]], groups, labels, 1)
        plt.scatter(i, j, c=colors[cLabel])
plt.show()
'''
print(classify([1,1], groups, labels, 3))
print(classify([5,5], groups, labels, 3))
print(classify([11,2], groups, labels, 3))
print(classify([2,13], groups, labels, 3))
'''