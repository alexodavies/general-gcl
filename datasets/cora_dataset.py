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
from torch_geometric.io import read_npz
import imageio
import wandb
# import osmnx as ox
from littleballoffur.exploration_sampling import MetropolisHastingsRandomWalkSampler, DiffusionSampler, ForestFireSampler
from sklearn.preprocessing import OneHotEncoder
# from ToyDatasets import *
import pickle
import zipfile
import wget
from networkx import community as comm


def download_cora(visualise = False):
    zip_url = "https://github.com/abojchevski/graph2gauss/raw/master/data/cora_ml.npz"

    start_dir = os.getcwd()
    print(os.getcwd(), os.listdir())
    os.chdir("original_datasets")

    if "cora" not in os.listdir():
        os.mkdir("cora")
        os.chdir("cora")
        _ = wget.download(zip_url)
    else:
        os.chdir("cora")
        # with zipfile.ZipFile("facebook_large.zip", 'r') as zip_ref:
        #     zip_ref.extractall(".")
        # os.remove("facebook_large.zip")
    # os.chdir("facebook_large")

    # edgelist = pd.read_csv("musae_facebook_edges.csv")

    edges = read_npz("cora_ml.npz")
    G = to_networkx(edges, to_undirected=True)

    node_classes = {n: edges.y[i].item() for i, n in enumerate(list(G.nodes()))}

    base_tensor = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])

    for node in list(G.nodes()):
        class_tensor = base_tensor
        class_tensor[node_classes[node]] = 1

        G.nodes[node]["attrs"] = class_tensor

    for edge in list(G.edges()):
        G.edges[edge]["attrs"] = torch.Tensor([1,0,0])

    # labels = pd.read_csv("musae_facebook_target.csv")
    # print(labels.head())
    # print(np.unique(labels["page_type"]))
    #
    # conversion_dict = {"company":       torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 0]),
    #                    "government":    torch.Tensor([2, 0, 0, 0, 0, 0, 0, 0, 0]),
    #                    "politician":    torch.Tensor([3, 0, 0, 0, 0, 0, 0, 0, 0]),
    #                    "tvshow":        torch.Tensor([4, 0, 0, 0, 0, 0, 0, 0, 0])}
    #
    #
    # graph = nx.Graph()
    # label_specific = labels["page_type"]
    # for col in labels["id"]:
    #     graph.add_node(int(col), attrs = conversion_dict[label_specific[col]]) # one_hot_embeddings[col].astype(float))
    # # print(edgelist)
    # sources = edgelist["id_1"].to_numpy().astype("int")
    # targets = edgelist["id_2"].to_numpy().astype("int")
    # #
    # # for i in range(sources):
    # #     source =
    #
    #
    # for i in range(sources.shape[0]):
    #     graph.add_edge(sources[i], targets[i], attr = torch.Tensor([1, 0, 0]))
    #
    # for node in list(graph.nodes(data=True)):
    #     data = node[1]
    #     if len(data) == 0:
    #         graph.remove_node(node[0])
    #
    # graph = nx.convert_node_labels_to_integers(graph)
    #
    # CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
    # CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    # graph = CGs[0]
    # graph = nx.convert_node_labels_to_integers(graph)
    # graph.remove_edges_from(nx.selfloop_edges(graph))
    #
    # with open("reddit-graph.npz", "wb") as f:
    #     pickle.dump(graph, f)
    #

    CGs = [G.subgraph(c) for c in nx.connected_components(G)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    graph = CGs[0]
    graph = nx.convert_node_labels_to_integers(graph)

    os.chdir(start_dir)
    # print(graph)
    # # quit()
    return graph

def ESWR(graph, n_graphs, size):
    print(f"Sampling {n_graphs} of size {size} from a {graph}")
    sampler = MetropolisHastingsRandomWalkSampler(number_of_nodes=size)
    # sampler = ForestFireSampler(number_of_nodes=size)
    graphs = [nx.convert_node_labels_to_integers(sampler.sample(graph)) for _ in tqdm(range(n_graphs))]

    return graphs



def get_cora_dataset(batch_size, num = 2000):
    fb_graph = download_cora()
    # print(fb_graph.nodes(data=True))
    nx_graph_list = ESWR(fb_graph, num, 48)


    loader = pyg.loader.DataLoader([pyg.utils.from_networkx(g, group_node_attrs=all, group_edge_attrs=all) for g in nx_graph_list],
                                              batch_size=batch_size)

    return loader

if __name__ == "__main__":
    # fb_graph = download_cora()
    # print(fb_graph.nodes(data=True))
    # graphs = ESWR(fb_graph, 200, 100)
    # G = download_cora()
    # print(G)
    loader = get_cora_dataset(512, num = 200)