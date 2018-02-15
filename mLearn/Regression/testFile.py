import pandas as pd
import quandl, math
import numpy as np
from statistics import mean

df = pd.DataFrame([[np.nan, 2, np.nan, 0], [3, 4, np.nan, 1],
                   [np.nan, np.nan, np.nan, 5]], columns=list('ABCD'))

X = np.array([1,2,3,4,5,6,7,8,9,10])

xmean = [mean(X) for y in X]

print(xmean)
print(X[:-3])
print(X[-3:])
print(df)
print(df.loc[0])
