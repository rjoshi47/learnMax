import numpy as np
from numpy import pi

a = np.arange(125).reshape(5, 5, 5)
#print(a.ndim)
#print(a.itemsize)

arr = np.array([1,2,3,4,5])
#print(arr)

twoDarr = np.array([(1,2), (3,4), (5,6), (2.1, 3.4)])
print(twoDarr)

colors = np.random.rand(5)
print(colors)

'''
print(twoDarr)

zeroArr = np.zeros((4,5), dtype=np.int32)
print(zeroArr)

oneArr = np.ones((4,5), dtype=np.int32)
print(oneArr)

seqArr = np.arange(4, 40, 2)
print(seqArr)

seqArr2 = np.linspace(0, 7, 8, dtype=int)
print(seqArr2)

sinArr = np.linspace(0, 2*pi, 100)
f = np.sin(sinArr)
print(f)


a = np.array([2,3,4,5])
b = np.array([3,-1,-2,2])
print(a+b)

small4 = a < 4
print(small4)
'''