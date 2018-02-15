import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, linear_model
import matplotlib.pyplot as plt
import pickle
from matplotlib import style


style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_CHANGE'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
df = df[['Adj. Close', 'HL_PCT', 'PCT_CHANGE', 'Adj. Volume']]

forecast_col = 'Adj. Close'
forecast_out = int(math.ceil(0.1*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out) #predicting next 10% values of Adj. Close

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:] # last forecastcount values
X = X[:-forecast_out] # till forecastcount values
df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = linear_model.LinearRegression()
clf.fit(X_train, y_train)

# saving classifier in file
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

read_clf = open('linearregression.pickle', 'rb')
clf = pickle.load(read_clf)

accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name # iloc gives values of whole row at given index
end_value = last_date.timestamp()
one_day = 86400 # seconds in a day
next_value = end_value + one_day

print([np.nan for _ in range(len(df.columns)-1)] + [forecast_set[0]])

for data in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_value)
    next_value += one_day
    # filling data in Forecast column only and others as nan
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [data]
    #df.loc represent the index which is date in this case

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()








