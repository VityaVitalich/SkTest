from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


@dataclass
class Node:
    name: str
    neighbors: List[str] = field(default_factory=lambda: [])
    special_info: str = None


class Graph:
    def __init__(self) -> None:
        self.all_nodes = {}

    def add_node(self, node_name: str, extra_info: Optional[str] = None) -> None:
        if node_name in self.all_nodes.keys():
            print("Warning! Node name is already present in current graph")

        self.all_nodes[node_name] = Node(node_name, special_info=extra_info)

    def add_edge(self, first_node: str, second_node: str) -> None:
        assert first_node in self.all_nodes.keys(), "no first node in graph"
        assert second_node in self.all_nodes.keys(), "no second node in graph"

        if second_node in self.all_nodes[first_node].neighbors:
            # print('edge is already presented')
            return

        self.all_nodes[first_node].neighbors.append(second_node)
        self.all_nodes[second_node].neighbors.append(first_node)

    def get_all_edges(self) -> Set[Tuple[str, str]]:
        all_edges = set()
        for name, node in self.all_nodes.items():
            for neighbor in node.neighbors:
                cur_edge = (name, neighbor)
                all_edges.add(cur_edge)

        return all_edges

    @staticmethod
    def generate_random(num_nodes: int = 10, p: int = 0.1) -> "Graph":
        random_names = [
            "Aleksandr",
            "Anna",
            "Dmitriy",
            "Yekaterina",
            "Ivan",
            "Mariya",
            "Nikolay",
            "Olga",
            "Pavel",
            "Svetlana",
            "Sergey",
            "Tatyana",
            "Aleksey",
            "Yelena",
            "Mikhail",
            "Egor",
            "Victor",
        ]

        g = Graph()
        unique_names = set()
        for i in range(num_nodes):
            cur_name = np.random.choice(random_names) + " 1"

            k = 2
            while cur_name in unique_names:
                cur_name = cur_name[:-1] + str(k)
                k += 1

            unique_names.add(cur_name)
            g.add_node(cur_name)

        for name in unique_names:
            for name2 in unique_names:
                if name == name2:
                    continue

                edge_exists = np.random.choice([0, 1], p=[1 - p, p])

                if edge_exists:
                    g.add_edge(name, name2)

        return g

    def draw_graph(self, options: Optional[Dict[str, Any]] = None) -> None:
        all_edges = self.get_all_edges()
        all_nodes = list(self.all_nodes.keys())

        G = nx.Graph()
        G.add_nodes_from(all_nodes)
        G.add_edges_from(all_edges)

        if not options:
            options = {
                "edgecolors": "tab:gray",
                "node_size": 1200,
                "alpha": 1,
                "font_size": 14,
                "node_color": "tab:red",
            }
        nx.draw(G, with_labels=True, **options)
        plt.show()
