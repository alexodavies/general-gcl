import numpy as np
import networkx as nx
import torch
import os
import torch_geometric as pyg
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import erdos_renyi_graph
import multiprocessing
from functools import partial
from tqdm.contrib.concurrent import process_map

def vis_from_pyg(data, filename = None):
    edges = data.edge_index.T.cpu().numpy()
    labels = data.x[:,0].cpu().numpy()

    g = nx.Graph()
    g.add_edges_from(edges)

    # dropped_nodes = np.ones(labels.shape[0]).astype(bool)
    for ilabel in range(labels.shape[0]):
        if ilabel not in np.unique(edges):
            g.add_node(ilabel)
    # labels = labels[dropped_nodes]

    fig, ax = plt.subplots(figsize = (6,6))

    pos = nx.kamada_kawai_layout(g)

    nx.draw_networkx_edges(g, pos = pos, ax = ax)
    nx.draw_networkx_nodes(g, pos = pos, node_color=labels, cmap="tab20",
                           vmin = 0, vmax = 20, ax = ax)

    ax.axis('off')

    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()

# def get_random_graph(size = 48):
#
#     rho = 0.05 + 0.15 * np.random.random()
#
#     # base_tensor = torch.Tensor([np.random.randint(1,9)])
#
#     G = nx.Graph()
#     for i in range(np.random.randint(low = 12, high=size)):
#         G.add_node(i, attr = torch.Tensor([1]))
#
#
#     for n1 in G.nodes():
#         for n2 in G.nodes():
#             if np.random.random() <= rho:
#                 G.add_edge(n1, n2, attr=torch.Tensor([1]))
#
#
#     return G, rho

def get_random_graph(size = 48):

    size = np.random.randint(low = 12, high = 128)

    rho = 0.05 + 0.25 * np.random.random()

    # base_tensor = torch.Tensor([np.random.randint(1,9)])

    edge_index = erdos_renyi_graph(size, rho)
    node_attr = torch.ones(size).reshape(-1,1)
    edge_attr = torch.ones(edge_index.shape[1]).reshape(-1,1)

    G = Data(node_attr, edge_index, edge_attr)

    return G, rho

def get_random_dataset(keep_target = False, num = 1000):

    # with multiprocessing.Pool(6) as pool:
    #     # r = list(tqdm.tqdm(p.imap(_foo, range(30)), total=30))
    #     nx_graph_list_rhos = list(tqdm(pool.imap(get_random_graph, range(num)), total = num))

    nx_graph_list_rhos = [get_random_graph() for _ in tqdm(range(num), leave=False)]
    datalist = [item[0] for item in nx_graph_list_rhos]
    rhos= [item[1] for item in nx_graph_list_rhos]
    # Ns = [graph.order() for graph in nx_graph_list]
    #
    # datalist = [pyg.utils.from_networkx(g, group_node_attrs=all, group_edge_attrs=all) for g in tqdm(nx_graph_list)]

    if keep_target:
        for idata, data in enumerate(datalist):
            data.y = torch.Tensor([rhos[idata]])
            datalist[idata] = data

    print(datalist)

    return datalist


# def get_random_dataset(keep_target = False, num = 1000):
#
#     # with multiprocessing.Pool(6) as pool:
#     #     # r = list(tqdm.tqdm(p.imap(_foo, range(30)), total=30))
#     #     nx_graph_list_rhos = list(tqdm(pool.imap(get_random_graph, range(num)), total = num))
#
#     nx_graph_list_rhos = [get_random_graph() for _ in tqdm(range(num), leave=False)]
#     nx_graph_list = [item[0] for item in nx_graph_list_rhos]
#     rhos= [item[1] for item in nx_graph_list_rhos]
#     Ns = [graph.order() for graph in nx_graph_list]
#
#     datalist = [pyg.utils.from_networkx(g, group_node_attrs=all, group_edge_attrs=all) for g in tqdm(nx_graph_list)]
#
#     if keep_target:
#         for idata, data in enumerate(datalist):
#             data.y = torch.Tensor([rhos[idata]])
#             datalist[idata] = data
#
#     return datalist


# def generate_random_graph(_, num):
#     return get_random_graph()
#
# def get_random_dataset(keep_target=False, num=1000):
#     with multiprocessing.Pool() as pool:
#         # Define a partial function to pass the 'num' argument
#         partial_generate = partial(generate_random_graph, num=num)
#
#         # Use process_map from tqdm to parallelize generation with progress bar
#         nx_graph_list_rhos = list(process_map(partial_generate, range(num), chunksize=10))
#
#     nx_graph_list = [item[0] for item in nx_graph_list_rhos]
#     rhos = [item[1] for item in nx_graph_list_rhos]
#     Ns = [graph.order() for graph in nx_graph_list]
#
#     datalist = [pyg.utils.from_networkx(g, group_node_attrs='all', group_edge_attrs='all') for g in tqdm(nx_graph_list)]
#
#     if keep_target:
#         for idata, data in enumerate(datalist):
#             data.y = torch.Tensor([rhos[idata]])
#             datalist[idata] = data
#
#     return datalist



class RandomDataset(InMemoryDataset):
    def __init__(self, root, stage="train", transform=None, pre_transform=None, pre_filter=None, num = 2000):
        self.num = num
        self.stage = stage
        self.stage_to_index = {"train":0,
                               "val":1,
                               "test":2}
        # _ = download_facebook()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.stage_to_index[self.stage]])

    @property
    def raw_file_names(self):
        return []


    @property
    def processed_file_names(self):
        return ['train.pt',
                'val.pt',
                'test.pt']


    def process(self):
        # Read data into huge `Data` list.

        if os.path.isfile(self.processed_paths[self.stage_to_index[self.stage]]):
            print("Random files exist")
            return

        data_list = get_random_dataset(num=self.num, keep_target=self.stage != "train")#get_fb_dataset(num=self.num)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # if self.stage != "train":
        #     for i, data in enumerate(data_list):
        #         vis_from_pyg(data, filename=self.root + f'/processed/{self.stage}-{i}.png')

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.stage_to_index[self.stage]])