from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')

dataset = {'k': [[1,2], [2,3], [3,1]], 'r': [[6,5], [7,7], [8,6]]}
new_featue = [3,4]

for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0], ii[1], s=100, color=i)

def k_nearest_neighbors(data, predit, k=3):
    if len(data) < k:
        distances = []
        for group in data:
            for features in data[group]:
                euclidean_distance = np.linalg.norm(np.array(features) - np.array(predit))
                distances.append([euclidean_distance, group])
        # sort distances array on euclidean_distance
        # get group identifier of first k distances
        votes = [i[1] for i in sorted(distances)[:k]]
        # counter(votes) for votes = ['r', 'r', 'r', 'k']
        # gives {'r': 3, 'k': 1}
        return Counter(votes).most_common(1)[0][0]


result = k_nearest_neighbors(dataset, new_featue, k=3)
plt.scatter(new_featue[0], new_featue[1], color=result)
plt.show()