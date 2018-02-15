from math import sqrt
import numpy as np
from collections import Counter
import pandas as pd
import random

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
        vote_result = Counter(votes).most_common(1)[0][0]
        confidence = Counter(votes).most_common(1)[0][1]/k
        return vote_result, confidence

df = pd.read_csv('C:\\Users\\rjoshi\\Documents\\ml\\breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.6
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

# the last column is the value of k either 2 or 4
# appending all columns except last to the empty list of train_set[2] and train_set[4]
for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0
for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        else:
            print(confidence)
        total += 1

print(correct/total)