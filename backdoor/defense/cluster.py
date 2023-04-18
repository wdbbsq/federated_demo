import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
from hdbscan import HDBSCAN
from backdoor.defense.pca import pca_of_gradients
from sklearn.preprocessing import StandardScaler

from utils.serialization import read_from_file
from utils.gradient import calc_vector_dist


def kmeans():
    # 要分类的数据点
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    # pyplot.scatter(x[:, 0], x[:, 1])
    # 把上面数据点分为两组（非监督学习）
    clf = KMeans(n_clusters=2)
    clf.fit(x)  # 分组

    centers = clf.cluster_centers_  # 两组数据点的中心点
    labels = clf.labels_  # 每个数据点所属分组
    print(centers)
    print(labels)

    for i in range(len(labels)):
        plt.scatter(x[i][0], x[i][1], c=('r' if labels[i] == 0 else 'b'))
    plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=100)

    # 预测
    predict = [[2, 1], [6, 9]]
    label = clf.predict(predict)
    for i in range(len(label)):
        plt.scatter(predict[i][0], predict[i][1], c=('r' if label[i] == 0 else 'b'), marker='x')
    plt.show()


def plot_cluster():
    kmeans()


def plot_in_2d(labels, x, title):
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = ['r', 'g', 'b', 'c', 'y', 'm', 'k']
    # colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
        plt.plot(x[labels == k, 0], x[labels == k, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)
    plt.title(title)
    plt.show()


def plot_in_3d(labels, x, title):
    # labels[3] = 0
    # labels[5] = 1
    # Black removed and is used for noise instead.
    colors = ['k', 'r', 'g', 'b', 'c', 'y', 'm']
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    length, _ = x.shape
    c = []
    for i in range(length):
        c.append(colors[labels[i] + 1])
    ax.scatter(x[0:, 0], x[0:, 1], x[0:, 2], c=c)

    plt.title(title)
    plt.show()


# clean
# obj = read_from_file('../logs/2023-03-20-14-58-06/7_cos_numpy')
# backdoor
obj = read_from_file('../logs/2023-03-20-14-58-06/29_cos_numpy')
# obj = read_from_file('../logs/2023-03-20-22-34-25/25_cos_numpy')
cos_list = obj['cos_list']

x = pca_of_gradients(cos_list, 2)

# kmeans
clf = KMeans(n_clusters=2)
clf.fit(x)  # 分组
print(calc_vector_dist(clf.cluster_centers_[0].reshape(1, -1),
                       clf.cluster_centers_[1].reshape(1, -1)))

centers = clf.cluster_centers_  # 两组数据点的中心点
labels = clf.labels_  # 每个数据点所属分组
print(centers)
print(labels)
plot_in_2d(labels, x, 'KMeans')
# plot_in_3d(labels, x, 'KMeans')

# hdbscan
# hdb = HDBSCAN(min_cluster_size=3).fit(x)
# labels = hdb.labels_
# outliers = hdb.outlier_scores_
# plot_in_2d(labels, x, 'DBSCAN')
# plot_in_3d(labels, x, 'DBSCAN')


# obj = read_from_file('./backdoor/logs/2023-03-20-14-58-06/26_cos_numpy')
# cos_list = obj['cos_list']
# x = pca_of_gradients(cos_list, 2)
#
# hdb = HDBSCAN(min_cluster_size=3, min_samples=1).fit(x)
# labels = hdb.labels_
# outliers = hdb.outlier_scores_
#
# plot_in_2d(labels, x)

