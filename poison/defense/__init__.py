from itertools import combinations
import networkx as nx
from networkx.algorithms.clique import find_cliques
from utils.gradient import calc_dist
import matplotlib.pyplot as plt


THRESHOLD = 6
STEP = 0.2
LAYER_NAME = 'fc2.0.weight'


def get_clean_updates(model_updates):
    n = len(model_updates)
    e = None
    while e is not None:
        graph = nx.Graph()
        for i, j in list(combinations(model_updates, 2)):
            graph.add_node(i[id])
            graph.add_node(j[id])
            dist = calc_dist(i['local_update'], j['local_update'], LAYER_NAME)
            print(dist)
            if dist < THRESHOLD:
                graph.add_edge(i['id'], j['id'])
        nx.draw(graph)
        plt.show()
        e = find_cliques(graph)

    return e
