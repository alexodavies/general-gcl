import numpy as np
import networkx as nx
import torch
import os
import torch_geometric as pyg
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.data import Data, InMemoryDataset



def get_tree_graph(max_nodes = 32):



    # base_tensor = torch.Tensor([np.random.randint(1,9)])

    n = np.random.randint(low=8, high = max_nodes)
    G = nx.random_tree(n = n)
    G_attr = nx.Graph()

    for i in range(G.order()):
        G_attr.add_node(i, attr = torch.Tensor([1]))

    for (n1, n2) in G.edges():
    # for n1 in G.nodes():
    #     for n2 in G.nodes():
        G_attr.add_edge(n1, n2, attr=torch.Tensor([1]))

    depth = nx.eccentricity(G, 0) / G.order()
    return G_attr, depth


def get_tree_dataset(keep_target = False, num = 1000):
    nx_graph_list_rhos = [get_tree_graph() for _ in tqdm(range(num), leave=False)]
    nx_graph_list = [item[0] for item in nx_graph_list_rhos]
    depths= [item[1] for item in nx_graph_list_rhos]
    Ns = [graph.order() for graph in nx_graph_list]

    datalist = [pyg.utils.from_networkx(g, group_node_attrs=all, group_edge_attrs=all) for g in tqdm(nx_graph_list)]

    if keep_target:
        for idata, data in enumerate(datalist):
            data.y = torch.Tensor([depths[idata]])
            datalist[idata] = data

    return datalist


class TreeDataset(InMemoryDataset):
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
            print("Tree files exist")
            return

        data_list = get_tree_dataset(num=self.num, keep_target=self.stage != "train")#get_fb_dataset(num=self.num)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # if self.stage != "train":
        #     for i, data in enumerate(data_list):
        #         vis_from_pyg(data, filename=self.root + f'/processed/{self.stage}-{i}.png')

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.stage_to_index[self.stage]])