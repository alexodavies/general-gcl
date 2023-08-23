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

def download_reddit(visualise = False):
    graph_url = "https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv"
    embedding_url = "http://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv"

    start_dir = os.getcwd()
    # for _ in range(3):
    #     os.chdir('../')
    print(os.getcwd(), os.listdir())
    os.chdir("original_datasets")

    if "reddit-graph.npz" in os.listdir():
        with open("reddit-graph.npz", "rb") as f:
            graph = pickle.load(f)
        os.chdir('../')
        return graph

    if "soc-redditHyperlinks-title.tsv" not in os.listdir():
        graph_data = wget.download(graph_url)
    if "web-redditEmbeddings-subreddits.csv" not in os.listdir():
        embedding_data = wget.download(embedding_url)


    embedding_column_names = ["COMPONENT", *[i for i in range(300)]]
    embeddings = pd.read_csv("web-redditEmbeddings-subreddits.csv", names=embedding_column_names).transpose()
    graph_data = pd.read_csv("soc-redditHyperlinks-title.tsv", sep = "\t")

    embeddings.columns = embeddings.iloc[0]
    embeddings = embeddings.drop(["COMPONENT"], axis = 0)


    graph = nx.Graph()

    for col in embeddings.columns:
        graph.add_node(col, attrs=embeddings[col].to_numpy().astype(float))

    sources = graph_data["SOURCE_SUBREDDIT"].to_numpy()
    targets = graph_data["TARGET_SUBREDDIT"].to_numpy()

    for i in range(sources.shape[0]):
        graph.add_edge(sources[i], targets[i])

    for node in list(graph.nodes(data=True)):
        data = node[1]
        if len(data) == 0:
            graph.remove_node(node[0])

    graph = nx.convert_node_labels_to_integers(graph)
    CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    graph = CGs[0]
    graph = nx.convert_node_labels_to_integers(graph)

    with open("reddit-graph.npz", "wb") as f:
        pickle.dump(graph, f)

    os.chdir(start_dir)

    return graph

def download_facebook(visualise = False):
    zip_url = "https://snap.stanford.edu/data/facebook_large.zip"

    start_dir = os.getcwd()
    print(os.getcwd(), os.listdir())
    os.chdir("original_datasets")

    if "facebook-graph.npz" in os.listdir():
        with open("facebook-graph.npz", "rb") as f:
            graph = pickle.load(f)
        os.chdir('../')
        return graph

    if "facebook_large" not in os.listdir():
        _ = wget.download(zip_url)
        with zipfile.ZipFile("facebook_large.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove("facebook_large.zip")
    os.chdir("facebook_large")

    edgelist = pd.read_csv("musae_facebook_edges.csv")

    labels = pd.read_csv("musae_facebook_target.csv")
    print(labels.head())
    print(np.unique(labels["page_type"]))

    conversion_dict = {"company":       torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 0]),
                       "government":    torch.Tensor([2, 0, 0, 0, 0, 0, 0, 0, 0]),
                       "politician":    torch.Tensor([3, 0, 0, 0, 0, 0, 0, 0, 0]),
                       "tvshow":        torch.Tensor([4, 0, 0, 0, 0, 0, 0, 0, 0])}


    graph = nx.Graph()
    label_specific = labels["page_type"]
    for col in labels["id"]:
        graph.add_node(int(col), attrs = conversion_dict[label_specific[col]]) # one_hot_embeddings[col].astype(float))
    # print(edgelist)
    sources = edgelist["id_1"].to_numpy().astype("int")
    targets = edgelist["id_2"].to_numpy().astype("int")
    #
    # for i in range(sources):
    #     source =


    for i in range(sources.shape[0]):
        graph.add_edge(sources[i], targets[i], attr = torch.Tensor([1, 0, 0]))

    for node in list(graph.nodes(data=True)):
        data = node[1]
        if len(data) == 0:
            graph.remove_node(node[0])

    graph = nx.convert_node_labels_to_integers(graph)

    CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    graph = CGs[0]
    graph = nx.convert_node_labels_to_integers(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    with open("reddit-graph.npz", "wb") as f:
        pickle.dump(graph, f)

    os.chdir(start_dir)
    print(graph)
    # quit()
    return graph

def ESWR(graph, n_graphs, size):
    sampler = MetropolisHastingsRandomWalkSampler(number_of_nodes=size)
    graphs = [nx.convert_node_labels_to_integers(sampler.sample(graph)) for _ in tqdm(range(n_graphs), leave=False)]

    return graphs

def get_fb_dataset(batch_size, num = 2000):
    fb_graph = download_facebook()
    # print(fb_graph.nodes(data=True))
    nx_graph_list = ESWR(fb_graph, num, 48)


    loader = pyg.loader.DataLoader([pyg.utils.from_networkx(g, group_node_attrs=all, group_edge_attrs=all) for g in nx_graph_list],
                                              batch_size=batch_size)

    return loader

if __name__ == "__main__":
    fb_graph = download_facebook()
    print(fb_graph.nodes(data=True))
    graphs = ESWR(fb_graph, 200, 100)