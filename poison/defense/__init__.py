from collections import defaultdict
from itertools import combinations
import networkx as nx
import numpy as np
from networkx.algorithms.clique import find_cliques
from utils.gradient import calc_dist
from utils.serialization import read_from_file
import matplotlib.pyplot as plt

THRESHOLD = 0.5
STEP = 0.02
LAYER_NAME = 'fc2.0.weight'


def plot_graph(graph):
    """
    画图
    https://blog.csdn.net/weixin_46348799/article/details/108169216
    """
    # colors = range(30)
    # options = {
    #     "node_size": 50,
    #     "font_color": "red",
    #     # "node_color": "#A0CBE2",
    #     # "edge_color": colors,
    #     "width": 2,
    #     "edge_cmap": plt.cm.Blues,
    # }

    # nx.draw(graph, with_labels=graph.nodes, pos=nx.spring_layout(graph))
    nx.draw(graph, with_labels=graph.nodes, pos=nx.circular_layout(graph))
    plt.show()


def clique(model_updates):
    global THRESHOLD
    n = len(model_updates)
    e = []
    while len(e) == 0:
        graph = nx.Graph()
        for i, j in list(combinations(model_updates, 2)):
            graph.add_node(i['id'])
            graph.add_node(j['id'])
            dist = calc_dist(i['local_update'], j['local_update'], LAYER_NAME)
            if dist < THRESHOLD:
                graph.add_edge(i['id'], j['id'])
        plot_graph(graph)
        it = find_cliques(graph)
        # 可能存在多个最大团结果，将包含元素最多的放到`e`中
        for cli in it:
            if len(cli) > len(e):
                e.clear()
                e = cli
        # 最大团中的元素小于客户端数量的一半，则阈值自增并重新计算
        if len(e) < (n / 2):
            e.clear()
            THRESHOLD = THRESHOLD + STEP
    return e


def krum(model_updates):
    """
    krum聚类算法
    """
    users_count = len(model_updates)
    non_malicious_count = users_count - model_updates
    minimal_error = 1e20
    minimal_error_index = -1

    distances = _krum_create_distances(model_updates)
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors[:non_malicious_count])
        if current_error < minimal_error:
            minimal_error = current_error
            minimal_error_index = user

    return model_updates[minimal_error_index]


def _krum_create_distances(users_grads):
    """
    计算距离
    """
    distances = defaultdict(dict)
    for i in range(len(users_grads)):
        for j in range(i):
            distances[i][j] = distances[j][i] = np.linalg.norm(users_grads[i] - users_grads[j])
    return distances


def trimmed_mean(model_updates):
    """
    裁剪均值算法
    """
    number_to_consider = int(model_updates.shape[0] - 1) - 1
    current_grads = np.empty((model_updates.shape[1],), model_updates.dtype)

    for i, param_across_users in enumerate(model_updates.T):
        med = np.median(param_across_users)
        good_vals = sorted(param_across_users - med, key=lambda x: abs(x))[:number_to_consider]
        current_grads[i] = np.mean(good_vals) + med
    return current_grads


def get_clean_updates(model_updates, method):
    if method == 'clique':
        return clique(model_updates)
    elif method == 'krum':
        return krum(model_updates)
    elif method == 'mean':
        return trimmed_mean(model_updates)
    else:
        raise NotImplementedError()

# 3
# local_updates = read_from_file('../logs/2023-03-26-16-35-28/18_dist')
# 6
# local_updates = read_from_file('../logs/2023-03-26-16-42-06/18_dist')
# 2
# local_updates = read_from_file('../logs/2023-03-26-16-47-53/18_dist')
# e = clique(local_updates)
