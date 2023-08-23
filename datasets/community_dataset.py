import numpy as np
import networkx as nx
import torch
import torch_geometric as pyg
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm

def get_community_graph(size = 48, proportions = [0.25, 0.25, 0.25, 0.25], P_intra = 0.5, P_inter=0.05):

    sizes = (np.array(proportions) * size).astype(int).tolist()#

    means = [torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 0]),
             torch.Tensor([2, 0, 0, 0, 0, 0, 0, 0, 0]),
             torch.Tensor([3, 0, 0, 0, 0, 0, 0, 0, 0]),
             torch.Tensor([4, 0, 0, 0, 0, 0, 0, 0, 0])]

    subgraphs = []
    counter = 0
    for i_size, size in enumerate(sizes):
        g = nx.Graph()

        for i in range(counter, counter + size):
            g.add_node(i, attrs= means[i_size]  )#np.random.randn(2) + means[i_size])

        counter += size
        subgraphs.append(g)

    for g in subgraphs:
        for n1 in g.nodes():
            for n2 in g.nodes():
                if np.random.random() <= P_intra:
                    g.add_edge(n1, n2, attr=torch.Tensor([1, 0, 0]))

    node_identifiers = [list(g.nodes()) for g in subgraphs]

    G = nx.Graph()
    for g in subgraphs:
        G = nx.compose(G, g)

    for ids_1 in node_identifiers:
        for ids_2 in node_identifiers:
            if ids_1 == ids_2:
                pass
            else:
                for n1 in ids_1:
                    for n2 in ids_2:
                        if np.random.random() <= P_inter:
                            G.add_edge(n1, n2, attr=torch.Tensor([0, 1, 0]))

    return G


def get_community_dataset(batch_size, num = 1000):
    nx_graph_list = [get_community_graph() for _ in tqdm(range(num), leave=False)]
    loader = pyg.loader.DataLoader([pyg.utils.from_networkx(g, group_node_attrs=all, group_edge_attrs=all) for g in tqdm(nx_graph_list)],
                                              batch_size=batch_size)
    return loader