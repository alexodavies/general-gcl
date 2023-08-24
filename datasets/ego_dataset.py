import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch_geometric as pyg
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

def get_deezer(batch_size, num = 49152):
    # zip_url = "https://snap.stanford.edu/data/deezer_ego_nets.zip"
    zip_url = "https://snap.stanford.edu/data/twitch_egos.zip"
    start_dir = os.getcwd()
    print(os.getcwd(), os.listdir())
    os.chdir("original_datasets")


    if "twitch_egos" not in os.listdir():
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

    loader = pyg.loader.DataLoader([pyg.utils.from_networkx(g, group_node_attrs=all, group_edge_attrs=all) for g in graphs],
                                              batch_size=batch_size)
    os.chdir(start_dir)
    return loader

    return loader

if __name__ == "__main__":
    graphs = get_deezer(num = 1000)
    print(graphs)