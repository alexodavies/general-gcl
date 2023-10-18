import os
import networkx as nx
import pandas as pd
import torch
from tqdm import tqdm
import torch_geometric as pyg
from torch_geometric.data import InMemoryDataset, Data
# import osmnx as ox
from littleballoffur.exploration_sampling import MetropolisHastingsRandomWalkSampler
# from ToyDatasets import *
import pickle
import zipfile
import wget
import matplotlib.pyplot as plt
import numpy as np
from utils import ESWR

import inspect
from littleballoffur.exploration_sampling import *
import littleballoffur.exploration_sampling as samplers

print(os.getcwd())
# from utils import vis_from_pyg


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

def download_reddit(visualise = False):
    graph_url = "https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv"
    embedding_url = "http://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv"

    start_dir = os.getcwd()
    # for _ in range(3):
    #     os.chdir('../')
    # print(os.getcwd(), os.listdir())
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
    # print(os.getcwd(), os.listdir())
    os.chdir("original_datasets")

    if "facebook-graph.npz" in os.listdir():
        with open("facebook-graph.npz", "rb") as f:
            graph = pickle.load(f)
        os.chdir('../')
        return graph
    # print(os.getcwd())

    if "facebook_large" not in os.listdir():
        print("Downloading FB graph")
        _ = wget.download(zip_url)
        with zipfile.ZipFile("facebook_large.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove("facebook_large.zip")
    os.chdir("facebook_large")

    edgelist = pd.read_csv("musae_facebook_edges.csv")

    labels = pd.read_csv("musae_facebook_target.csv")
    # print(labels.head())
    # print(np.unique(labels["page_type"]))

    conversion_dict = {"company":       torch.Tensor([1]), #, 0, 0, 0, 0, 0, 0, 0, 0]),
                       "government":    torch.Tensor([2]), #, 0, 0, 0, 0, 0, 0, 0, 0]),
                       "politician":    torch.Tensor([3]), #, 0, 0, 0, 0, 0, 0, 0, 0]),
                       "tvshow":        torch.Tensor([4])} #, 0, 0, 0, 0, 0, 0, 0, 0]),


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
        graph.add_edge(sources[i], targets[i], attr = torch.Tensor([1]))

    for node in list(graph.nodes(data=True)):
        data = node[1]
        if len(data) == 0:
            graph.remove_node(node[0])

    graph = nx.convert_node_labels_to_integers(graph)
    # print(f"Facebook {graph}")

    CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    graph = CGs[0]
    graph = nx.convert_node_labels_to_integers(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    with open("reddit-graph.npz", "wb") as f:
        pickle.dump(graph, f)

    os.chdir(start_dir)
    # print(graph)
    # quit()
    return graph

# def ESWR(graph, n_graphs, size):
#
#     # possible_samplers = inspect.getmembers(samplers, inspect.isclass)
#     #
#     # possible_samplers = [item[1] for item in possible_samplers]
#     # possible_samplers = [MetropolisHastingsRandomWalkSampler, DiffusionSampler, DepthFirstSearchSampler]
#     # # selected_sampler = possible_samplers[np.random.randint(len(possible_samplers))]
#     #
#     #
#     # print(f"Sampling {n_graphs} graphs from {graph}")
#     # graphs = []
#     # for i in tqdm(range(n_graphs), leave = False):
#     #     selected_sampler = possible_samplers[np.random.randint(len(possible_samplers))]
#     #     sampler = selected_sampler(number_of_nodes=np.random.randint(12, 48))
#     #     graphs.append(nx.convert_node_labels_to_integers(sampler.sample(graph)))
#     # sampler = selected_sampler(number_of_nodes=np.random.randint(12, 36))
#     sampler = MetropolisHastingsRandomWalkSampler(48)
#     graphs = [nx.convert_node_labels_to_integers(sampler.sample(graph)) for i in tqdm(range(n_graphs))]
#
#     return graphs

def get_fb_dataset(num = 2000, targets = False):
    fb_graph = download_facebook()
    # print(fb_graph.nodes(data=True))
    nx_graph_list = ESWR(fb_graph, num, 48)


    # loader = pyg.loader.DataLoader([pyg.utils.from_networkx(g, group_node_attrs=all, group_edge_attrs=all) for g in nx_graph_list],
    #                                           batch_size=batch_size)
    data_objects = [pyg.utils.from_networkx(g, group_node_attrs=all, group_edge_attrs=all) for g in nx_graph_list]



    for i_data, data in enumerate(tqdm(data_objects, desc="Calculating clustering values for FB", leave=False)):
        if targets:
            data.y = torch.tensor(nx.average_clustering(nx_graph_list[i_data])) # None # torch.Tensor([[0,0]])
        else:
            data.y = torch.tensor([1.])

    return data_objects# loader

class FacebookDataset(InMemoryDataset):
    def __init__(self, root, stage="train", transform=None, pre_transform=None, pre_filter=None, num = 2000):
        self.num = num
        self.stage = stage
        self.stage_to_index = {"train":0,
                               "val":1,
                               "test":2}
        _ = download_facebook()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.stage_to_index[self.stage]])

    @property
    def raw_file_names(self):
        return ['musae_facebook_edges.csv',
                'musae_facebook_target.csv']


    @property
    def processed_file_names(self):
        return ['train.pt',
                'val.pt',
                'test.pt']


    def process(self):
        # Read data into huge `Data` list.

        if os.path.isfile(self.processed_paths[self.stage_to_index[self.stage]]):
            print("Facebook files exist")
            return

        data_list = get_fb_dataset(num=self.num, targets=self.stage != "train")

        if self.stage == "train":
            print("Found stage train, dropping targets")
            new_data_list = []
            for i, item in enumerate(data_list):
                n_nodes, n_edges = item.x.shape[0], item.edge_index.shape[1]

                data = Data(x = torch.ones(n_nodes).to(torch.int).reshape((-1, 1)),
                            edge_index=item.edge_index,
                            edge_attr=torch.ones(n_edges).to(torch.int).reshape((-1,1)),
                            y = None)

                # data = Data(x = item.x[:,0].reshape((-1, 1)), edge_index=item.edge_index,
                #             edge_attr=item.edge_attr, y = None)
                # print(f"Train x shape {data.x.shape}, edge index {data.edge_index.shape}, edge attr {data.edge_attr.shape}")
                # print(data)
                # vis_from_pyg(data, filename=self.root + '/processed/' + i + '.png')
                new_data_list.append(data)
            data_list = new_data_list
        else:
            new_data_list = []
            for i, item in enumerate(data_list):
                n_nodes, n_edges = item.x.shape[0], item.edge_index.shape[1]


                data = Data(x = item.x,# torch.ones(n_nodes).to(torch.int).reshape((-1, 1)),
                            edge_index=item.edge_index,
                            edge_attr=torch.ones(n_edges).to(torch.int).reshape((-1,1)),
                            y = item.y)

                # data = Data(x = item.x[:,0].reshape((-1, 1)), edge_index=item.edge_index,
                #             edge_attr=item.edge_attr, y = item.y)
                # print(f"Val x shape {data.x.shape}, edge index {data.edge_index.shape}")
                # print(data)
                # vis_from_pyg(data, filename=self.root + '/processed/' + i + '.png')
                new_data_list.append(data)
            data_list = new_data_list


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
    os.chdir('../')
    dataset = FacebookDataset(os.getcwd()+'/original_datasets/'+'facebook_large', stage = "val")