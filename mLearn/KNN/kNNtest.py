from numpy import *
import operator

dset = array([[0, 0], [10, 0], [10, 10], [0, 10]])
labels = ['A', 'B', 'C', 'D']

#dset = array([[1,2], [3,4], [7,8], [1,2], [3,4], [7,8]], dtype=int)
print(dset)

difmat = tile([2,11], (dset.shape[0],1)) - dset
sqdifmat = difmat**2
print(difmat)
print(sqdifmat)

sqdis = sqdifmat.sum(axis=1)
print(sqdis)
disort = sqdis.argsort()
print(disort)

labelCountDict = {}
visitedLabels = []
for i in range(1):
    cLabel = labels[disort[i]]
    labelCountDict[cLabel] = labelCountDict.get(cLabel,0) + 1

sortedCount = sorted(labelCountDict.items(),
                     key=operator.itemgetter(1), reverse=True)
print('ddd')
print(sortedCount)
print(sortedCount[0][0])
print(labelCountDict)
labelCountDict['E']=labelCountDict.get('E',4)
print(labelCountDict)
