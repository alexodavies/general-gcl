import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch_geometric as pyg
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import to_networkx
import imageio
import wandb
# import osmnx as ox
from littleballoffur.exploration_sampling import MetropolisHastingsRandomWalkSampler
from sklearn.preprocessing import OneHotEncoder
# from ToyDatasets import *
import pickle
import zipfile
import wget
from networkx import community as comm

def get_deezer(num = 49152):
    # zip_url = "https://snap.stanford.edu/data/deezer_ego_nets.zip"
    zip_url = "https://snap.stanford.edu/data/twitch_egos.zip"
    start_dir = os.getcwd()
    # print(os.getcwd(), os.listdir())
    os.chdir("original_datasets")


    if "twitch_egos" not in os.listdir():
        print("Downloading Twitch Egos")
        _ = wget.download(zip_url)
        with zipfile.ZipFile("twitch_egos.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove("twitch_egos.zip")
    os.chdir("twitch_egos")


    with open("twitch_edges.json", "r") as f:
        all_edges = json.load(f)
    graph_ids = list(all_edges.keys())

    graphs = []

    for id in graph_ids[:num]:
        edges = all_edges[id]

        g = nx.Graph()

        nodes = np.unique(edges).tolist()

        for node in nodes:
            g.add_node(node, attr = torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 0]))

        for edge in edges:
            g.add_edge(edge[0], edge[1], attr=torch.Tensor([1, 0, 0]))
        graphs.append(g)

    # loader = pyg.loader.DataLoader([pyg.utils.from_networkx(g, group_node_attrs=all, group_edge_attrs=all) for g in graphs],
    #                                           batch_size=batch_size)
    os.chdir(start_dir)
    # return loader
    data_objects = [pyg.utils.from_networkx(g, group_node_attrs=all, group_edge_attrs=all) for g in graphs]
    for data in data_objects:
        data.y = None # torch.Tensor([[0,0]])

    return  data_objects# loader

class EgoDataset(InMemoryDataset):
    def __init__(self, root, stage = "train", transform=None, pre_transform=None, pre_filter=None, num = 2000):
        self.num = num
        self.stage = stage
        self.stage_to_index = {"train":0,
                               "val":1,
                               "test":2}
        _ = get_deezer()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.stage_to_index[self.stage]])

    @property
    def raw_file_names(self):
        return ['twitch_edges.json',
                'twitch_target.json']

    @property
    def processed_file_names(self):
        return ['train.pt',
                'val.pt',
                'test.pt']


    def process(self):
        # Read data into huge `Data` list.
        data_list = get_deezer(num=self.num)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.stage_to_index[self.stage]])


if __name__ == "__main__":
    # fb_graph = download_cora()
    # print(fb_graph.nodes(data=True))
    # graphs = ESWR(fb_graph, 200, 100)
    # G = download_cora()
    # print(G)
    dataset = EgoDataset(os.getcwd()+'/original_datasets/'+'twitch_egos')