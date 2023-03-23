import numpy as np
from utils.serialization import read_from_file
from hdbscan import HDBSCAN
from backdoor.defense import pca_of_gradients

GAP = 0.6


# time_str = '2023-03-21-08-16-19'
# adversary_num = 4

# time_str = '2023-03-20-22-34-25'
# adversary_num = 2

# time_str = '2023-03-20-22-34-25'
time_str = '2023-03-21-16-58-44'
adversary_num = 1

acc = []
for i in range(20, 30):
    obj = read_from_file(f'../logs/{time_str}/{i}_cos_numpy')
    cos_list = obj['cos_list']
    x = pca_of_gradients(cos_list, 3)

    hdb = HDBSCAN(min_cluster_size=3, min_samples=1).fit(x)
    labels = hdb.labels_
    outliers = hdb.outlier_scores_
    tp = 0
    for x in outliers[:adversary_num]:
        if x > GAP:
            tp += 1
    tn = 0
    for x in outliers[adversary_num:]:
        if x <= GAP:
            tn += 1
    acc.append((tp + tn) / 20)
print(acc)
print(np.mean(acc))

a = [0.95, 0.95, 0.9, 1.0, 1.0, 0.9, 0.95, 0.9, 0.95, 1.0]
print(np.mean(a))
