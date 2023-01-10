import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
from typing import List, Tuple, Callable
import seaborn
import random
from dataclasses import dataclass
from itertools import islice

nodes = 'graph_data/nodes.csv'
edges = 'graph_data/edges.csv'


def load_graph() -> nx.DiGraph:
    nodes_df = pd.read_csv(nodes).dropna()
    edges_df = pd.read_csv(edges).dropna()
    nodes_tuple = tuple(zip(range(len(nodes_df)),
                            nodes_df['type'],
                            nodes_df['x'],
                            nodes_df['y']))
    edges_tuple = tuple(zip(range(len(edges_df)),
                            edges_df['node1'],
                            edges_df['node2'],
                            edges_df['weight'],
                            edges_df['min_type']))
    graph = nx.DiGraph()
    for node in nodes_tuple:
        graph.add_node(node[0], type=node[1], pos=(node[2], node[3]))
    for edge in edges_tuple:
        graph.add_edge(edge[1], edge[2],
                       index=edge[0],
                       dist=edge[3],
                       type=edge[4])
        graph.add_edge(edge[2], edge[1],
                       index=-edge[0],
                       dist=edge[3],
                       type=edge[4])
    return graph


def calc_price(edge_type: int, dist: float, payload: float):
    if payload != 0:
        return cfg.type_info[edge_type].km_price * dist * (payload // cfg.type_info[edge_type].capacity + 1)
    else:
        return 0


def draw_price_function(edge_type, dist, max_payload):
    edge_payload = np.arange(-0.01, max_payload, 0.1)
    edge_price = np.array(calc_price(edge_type, dist, edge_payload))

    plt.xlabel('Нагрузка')
    plt.ylabel('Цена')

    plt.plot(edge_payload, edge_price)


def draw_graph(graph: nx.Graph, what: List[str] = None, arrowstyle='-'):
    edge_palette = seaborn.color_palette('pastel', 3)
    node_palette = seaborn.color_palette('muted', 3)
    pos = nx.get_node_attributes(graph, 'pos')
    nodes_types = nx.get_node_attributes(graph, 'type')
    edges_types = nx.get_edge_attributes(graph, 'type')
    if what is None or 'edges' in what:
        nx.draw_networkx_edges(graph, pos=pos,
                               edge_color=list(map(lambda x: edge_palette[x], edges_types.values())),
                               width=list(map(lambda x: (x + 1) * 1.5, edges_types.values())),
                               alpha=list(map(lambda x: (x + 1) / 3, edges_types.values())),
                               arrows=True,
                               arrowstyle=arrowstyle)
    if what is None or 'nodes' in what:
        nx.draw_networkx_nodes(graph, pos=pos,
                               node_color=list(map(lambda x: node_palette[x], nodes_types.values())),
                               node_size=list(map(lambda x: (x + 1) * 20, nodes_types.values())))
    if what is None or 'labels' in what:
        nx.draw_networkx_labels(graph, pos=pos,
                                alpha=0.5)


def edges_from_path(p: List[int]) -> List[Tuple[int, int]]:
    result = []
    for j in range(len(p) - 1):
        result.append((p[j], p[j + 1]))
    return result


def generate_orders(graph: nx.Graph, n: int) -> List[Tuple[int, int]]:
    orders = []
    for i in range(n):
        orders.append(tuple(random.sample(graph.nodes(), 2)))
    return orders


def draw_payload(graph: nx.Graph, used_edges_payload):
    pos = nx.get_node_attributes(graph, 'pos')
    edge_colors = {}
    max_edges_payload = max(used_edges_payload.values())
    for edge, payload in used_edges_payload.items():
        edge_colors[edge] = (0, 0, 0, payload / max_edges_payload)
    draw_graph(graph, ['nodes'])
    nx.draw_networkx_edges(graph, pos=pos, arrows=True, width=2, edge_color=edge_colors.values(),
                           edgelist=edge_colors.keys(), connectionstyle='arc3, rad = 0.05', arrowstyle='->')


def calc_total_price(graph, used_edges_payload):
    edges_prices = {}
    for edge, payload in used_edges_payload.items():
        edge_attributes = graph[edge[0]][edge[1]]
        price = calc_price(edge_attributes['type'], edge_attributes['dist'], payload)
        edges_prices[edge] = price
    total_price = sum(edges_prices.values())
    return edges_prices, total_price


@dataclass
class Result:
    orders: List[Tuple[int, int]]
    paths: List[List[int]]
    edges_payload: dict
    edges_prices: dict
    total_price: float
    time: float


def k_shortest(graph, source, target, k, weight=None):
    return list(
        islice(nx.shortest_simple_paths(graph, source, target, weight=weight),
               k)
    )
#
# def get_result(calc_paths_callback: Callable, ):
#     t0 = time.time()
#     _, paths, edges_payload = calc_trivial_paths(orders)
#     edges_prices, total_price = u.calc_total_price(G, edges_payload)
#     results.append(u.Result(orders, paths, edges_payload, edges_prices, total_price, time.time() - t0))
