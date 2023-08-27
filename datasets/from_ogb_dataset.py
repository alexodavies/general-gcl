import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch_geometric as pyg
from torch_geometric.data import InMemoryDataset, Data
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


class FromOGBDataset(InMemoryDataset):
    def __init__(self, root, ogb_dataset, transform=None, pre_transform=None, pre_filter=None):
        self.ogb_dataset = ogb_dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['dummy.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']


    def process(self):
        # Read data into huge `Data` list.
        data_list = self.ogb_dataset# get_fb_dataset(num=self.num)

        new_data_list = []
        for item in data_list:
            data = Data(x = item.x, edge_index=item.edge_index,
                        edge_attr=item.edge_attr, y = None)
            new_data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(new_data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    # fb_graph = download_cora()
    # print(fb_graph.nodes(data=True))
    # graphs = ESWR(fb_graph, 200, 100)
    # G = download_cora()
    # print(G)
    dataset = FacebookDataset(os.getcwd()+'/original_datasets/'+'facebook_large')