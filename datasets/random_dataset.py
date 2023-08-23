import numpy as np
import networkx as nx
import torch
import torch_geometric as pyg
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm

def get_random_graph(size = 48):

    rho = 0.4 * np.random.random()

    base_tensor = torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 0])

    G = nx.Graph()
    for i in range(size):
        G.add_node(i, attr = base_tensor)


    for n1 in G.nodes():
        for n2 in G.nodes():
            if np.random.random() <= rho:
                G.add_edge(n1, n2, attr=torch.Tensor([1, 0, 0]))


    return G


def get_random_dataset(batch_size, num = 1000):
    nx_graph_list = [get_random_graph() for _ in tqdm(range(num), leave=False)]
    loader = pyg.loader.DataLoader([pyg.utils.from_networkx(g, group_node_attrs=all, group_edge_attrs=all) for g in tqdm(nx_graph_list)],
                                              batch_size=batch_size)
    return loader