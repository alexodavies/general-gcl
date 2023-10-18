import json
import os
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg
from torch_geometric.data import InMemoryDataset
# import osmnx as ox
# from ToyDatasets import *
import zipfile
import wget
# from utils import vis_from_pyg
import matplotlib.pyplot as plt
from tqdm import tqdm

def vis_from_pyg(data, filename = None):
    edges = data.edge_index.T.cpu().numpy()
    labels = data.x[:,0].cpu().numpy()

    g = nx.Graph()
    g.add_edges_from(edges)

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

def get_twitch(num = 49152, include_targets = False):
    # zip_url = "https://snap.stanford.edu/data/deezer_ego_nets.zip"
    print("Processing twitch egos dataset")
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

    if include_targets:
        twitch_targets = pd.read_csv("twitch_target.csv")
        ids, targets = twitch_targets["id"], twitch_targets["target"]
        id_to_target = {ids[i]:targets[i] for i in range(len(ids))}


    graph_ids = list(all_edges.keys())

    graphs = []

    print("Entering ego processing loop")
    for id in tqdm(graph_ids[:num], leave = False):
        edges = all_edges[id]

        g = nx.Graph()

        nodes = np.unique(edges).tolist()

        for node in nodes:
            g.add_node(node, attr = torch.Tensor([1])) #, 0, 0, 0, 0, 0, 0, 0, 0]))

        for edge in edges:
            g.add_edge(edge[0], edge[1], attr=torch.Tensor([1]))
        graphs.append(g)

    # loader = pyg.loader.DataLoader([pyg.utils.from_networkx(g, group_node_attrs=all, group_edge_attrs=all) for g in graphs],
    #                                           batch_size=batch_size)
    os.chdir(start_dir)
    # return loader
    data_objects = [pyg.utils.from_networkx(g, group_node_attrs=all, group_edge_attrs=all) for g in graphs]


    for id, data in enumerate(data_objects):
        if include_targets:
            data.y = id_to_target[id]
        else:
            data.y = None # torch.Tensor([[0,0]])

    return  data_objects# loader

class EgoDataset(InMemoryDataset):
    def __init__(self, root, stage = "train", transform=None, pre_transform=None, pre_filter=None, num = 2000):
        self.num = num
        self.stage = stage
        self.stage_to_index = {"train":0,
                               "val":1,
                               "test":2}


        _ = get_twitch(num = 1)
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

        if os.path.isfile(self.processed_paths[self.stage_to_index[self.stage]]):
            print("Ego files exist")
            return
        print(f"Aiming for {self.num} graphs")
        data_list = get_twitch(num=self.num, include_targets =self.stage != "train")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # if self.stage != "train":
        #     for i, data in enumerate(data_list):
        #         vis_from_pyg(data, filename=self.root + f'/processed/{self.stage}-{i}.png')

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.stage_to_index[self.stage]])


if __name__ == "__main__":
    # fb_graph = download_cora()
    # print(fb_graph.nodes(data=True))
    # graphs = ESWR(fb_graph, 200, 100)
    # G = download_cora()
    # print(G)
    dataset = EgoDataset(os.getcwd()+'/original_datasets/'+'twitch_egos')