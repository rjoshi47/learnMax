import matplotlib
from numpy import *
import matplotlib.pyplot as plt

twoDarr = array([(1,2), (3,4), (5,6), (2.1, 3.4)])
colors = [1,2,3,4]
x = arange(0, 5, 0.1)
y = sin(x)
plt.subplot(251)
plt.scatter(twoDarr[:,0],twoDarr[:,1], c=colors)
plt.scatter(twoDarr[:,1],twoDarr[:,1], c=colors)
#plt.plot(x, y, linewidth=2.0, color='r')

#plt.subplot(252)
#plt.plot(x, y, linewidth=2.0, color='r')
#plt.axis([0,10,0,20])
plt.show()