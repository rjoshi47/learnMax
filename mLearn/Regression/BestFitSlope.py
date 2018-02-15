from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

# create dataset to test coefficient of goodness
def create_dataset(n, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(n):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope_intercept(xs: object, ys: object) -> object:
    m = ((mean(xs)*mean(ys) - mean(xs*ys))
            /(mean(xs)*mean(xs) - mean(xs**2)))
    b = mean(ys) - m*mean(xs)
    return  m, b

def squared_error(ys_orig, ys_reg_line):
    return sum((ys_reg_line-ys_orig)**2)

def coefficient_of_goodness(ys_orig, ys_reg_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig] # each element has same mean value
    sq_error_regr = squared_error(ys_orig, ys_reg_line)
    sq_error_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (sq_error_regr/sq_error_mean)

# coefficient_of_goodness increases with decrease in variance
xs, ys = create_dataset(40, 10, 2, correlation='pos')

m, b = best_fit_slope_intercept(xs, ys)
regression_line = [(m*x)+b for x in xs]
print(coefficient_of_goodness(ys, regression_line))

plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()