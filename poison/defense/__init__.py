from itertools import combinations
import networkx as nx
from networkx.algorithms.clique import find_cliques
from utils.gradient import calc_dist
from utils.serialization import read_from_file
import matplotlib.pyplot as plt

THRESHOLD = 0.5
STEP = 0.02
LAYER_NAME = 'fc2.0.weight'


def plot_graph(graph):
    """
    https://blog.csdn.net/weixin_46348799/article/details/108169216
    """
    colors = range(30)
    # options = {
    #     "node_size": 50,
    #     "font_color": "red",
    #     # "node_color": "#A0CBE2",
    #     # "edge_color": colors,
    #     "width": 2,
    #     "edge_cmap": plt.cm.Blues,
    # }
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
                print(dist)
                graph.add_edge(i['id'], j['id'])
        plot_graph(graph)
        it = find_cliques(graph)
        for cli in it:
            if len(cli) > len(e):
                e.clear()
                e = cli
        # if len(e) < n / 2:
        #     e.clear()
        #     THRESHOLD = THRESHOLD + STEP
    print(e)
    return e


def krum():
    print(1)


def trimmed_mean():
    print(1)


def get_clean_updates(model_updates, method):
    if method == 'clique':
        return clique(model_updates)
    elif method == 'krum':
        krum()
    elif method == 'mean':
        trimmed_mean()
    else:
        raise NotImplementedError()


# 3
# local_updates = read_from_file('../logs/2023-03-26-16-35-28/18_dist')
# 6
# local_updates = read_from_file('../logs/2023-03-26-16-42-06/18_dist')
# 2
# local_updates = read_from_file('../logs/2023-03-26-16-47-53/18_dist')
# e = clique(local_updates)

