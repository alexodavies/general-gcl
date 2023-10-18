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
import gzip
import shutil
import inspect
from utils import ESWR
from littleballoffur.exploration_sampling import *
import littleballoffur.exploration_sampling as samplers

print(os.getcwd())
# from utils import vis_from_pyg

def four_cycles(g):
    """
    Returns the number of 4-cycles in a graph, normalised by the number of nodes
    """
    cycles = nx.simple_cycles(g, 4)
    return len(list(cycles)) / g.order()

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


def download_roads(visualise = False):
    zip_url = "https://snap.stanford.edu/data/roadNet-PA.txt.gz"

    start_dir = os.getcwd()
    # print(os.getcwd(), os.listdir())
    os.chdir("original_datasets")


    if "roads" not in os.listdir():
        print("Downloading roads graph")
        os.mkdir("roads")
        os.chdir("roads")

        _ = wget.download(zip_url)

        with gzip.open('roadNet-PA.txt.gz', 'rb') as f_in:
            with open('roadNet-PA.txt', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # with zipfile.ZipFile("facebook_large.zip", 'r') as zip_ref:
        #     zip_ref.extractall(".")
        os.remove("roadNet-PA.txt.gz")
    else:
        os.chdir("roads")

        if "road-graph.npz" in os.listdir():
            with open("road-graph.npz", "rb") as f:
                graph = pickle.load(f)
            os.chdir('../')
            return graph
    # print(os.getcwd())


    # os.chdir("facebook_large")

    edgelist = pd.read_csv("roadNet-PA.txt", delimiter = "\t", header=3)

    graph = nx.Graph()
    # for col in labels["id"]:
    #     graph.add_node(int(col), attrs = torch.Tensor([1]))#conversion_dict[label_specific[col]]) # one_hot_embeddings[col].astype(float))
    # print(edgelist)
    sources = edgelist["# FromNodeId"].to_numpy().astype("int")
    targets = edgelist["ToNodeId"].to_numpy().astype("int")
    #
    # for i in range(sources):
    #     source =
    edges_to_add = [(sources[i], targets[i]) for i in tqdm(range(sources.shape[0]))]
    # attr_to_add = [torch.Tensor([1]) for i in tqdm(range(sources.shape[0]))]

    graph.add_edges_from(edges_to_add)

    del edges_to_add

    # attr_to_add = [torch.Tensor([1]) for i in range(sources.shape[0])]

    # for i in range(sources.shape[0]):
    #     graph.add_edge(sources[i], targets[i], attr = torch.Tensor([1]))
    # for node in graph.nodes():
    #     graph.nodes[node]["attr"] = torch.Tensor([1])

    # for node in list(graph.nodes(data=True)):
    #     data = node[1]
    #     if len(data) == 0:
    #         graph.remove_node(node[0])

    graph = nx.convert_node_labels_to_integers(graph)

    CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    graph = CGs[0]
    graph = nx.convert_node_labels_to_integers(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    with open("road_graph.npz", "wb") as f:
        pickle.dump(graph, f)

    os.chdir(start_dir)
    # print(graph)
    # quit()
    return graph


def individual_sample(graph):
    sampler = DiffusionSampler(number_of_nodes=np.random.randint(12, 48))
    return nx.convert_node_labels_to_integers(sampler.sample(graph))

def add_attrs_to_graph(g):
    # for node in graph.nodes():
    #     graph.nodes[node]["attr"] = torch.Tensor([1])
    # print({n:torch.Tensor([1]).to(torch.int) for n in list(g.nodes())})
    # print({e:torch.Tensor([1]).to(torch.int) for e in list(g.edges())})
    # nx.set_node_attributes(g, {n:torch.Tensor([1]).to(torch.int) for n in list(g.nodes())})
    # nx.set_edge_attributes(g, {e:torch.Tensor([1]).to(torch.int) for e in list(g.edges())})

    # for node in g.nodes():
    #     g.nodes[node]["attr"] = torch.Tensor([1])
    #
    # for edge in g.edges():
    #     g.edges[node]["attr"] = torch.Tensor([1])
    dummy_label = torch.Tensor([1])
    nx.set_edge_attributes(g, dummy_label, "attrs")
    nx.set_node_attributes(g, dummy_label, "attrs")

    # dummy_label.append(torch.Tensor([1]))

    return g

# def ESWR(graph, n_graphs, size):
    #
    # # possible_samplers = inspect.getmembers(samplers, inspect.isclass)
    # #
    # # possible_samplers = [item[1] for item in possible_samplers]
    # # possible_samplers = [MetropolisHastingsRandomWalkSampler, DiffusionSampler, DepthFirstSearchSampler]
    # # # selected_sampler = possible_samplers[np.random.randint(len(possible_samplers))]
    # #
    # #
    # # print(f"Sampling {n_graphs} graphs from {graph}")
    # # graphs = []
    # # for i in tqdm(range(n_graphs), leave = False):
    # #     selected_sampler = possible_samplers[np.random.randint(len(possible_samplers))]
    # #     sampler = selected_sampler(number_of_nodes=np.random.randint(12, 48))
    # #     graphs.append(nx.convert_node_labels_to_integers(sampler.sample(graph)))
    # # sampler = selected_sampler(number_of_nodes=np.random.randint(12, 36))
    # sampler = MetropolisHastingsRandomWalkSampler(96)
    # graphs = [nx.convert_node_labels_to_integers(sampler.sample(graph)) for i in tqdm(range(n_graphs))]
    #
    # return graphs

def get_road_dataset(num = 2000, targets = False):
    fb_graph = download_roads()
    # print(fb_graph.nodes(data=True))
    nx_graph_list = ESWR(fb_graph, num, 48)
    nx_graph_list = [add_attrs_to_graph(g) for g in nx_graph_list]


    # loader = pyg.loader.DataLoader([pyg.utils.from_networkx(g, group_node_attrs=all, group_edge_attrs=all) for g in nx_graph_list],
    #                                           batch_size=batch_size)
    data_objects = [pyg.utils.from_networkx(g, group_node_attrs=all, group_edge_attrs=all) for g in nx_graph_list]

    for i, data in enumerate(tqdm(data_objects, desc="Calculating diameter values for roads", leave=False)):
        if targets:
            data.y = torch.tensor(nx.diameter(nx_graph_list[i]) / nx_graph_list[i].order()) #None # torch.Tensor([[0,0]])
        else:
            data.y = torch.tensor([1])
        # print(data.edge_index, data.x, data.edge_attr)
        # data.edge_attr = torch.ones(data.edge_index.shape[1]).to(torch.int).reshape((-1, 1))
        # data.x = torch.ones(data.num_nodes).to(torch.int).reshape((-1, 1))
        # data_objects[i] = data

        data.edge_attr = data.edge_attr[:,0].reshape(-1,1)

        # data.edge_attr = torch.ones(data.edge_index.shape[1]).to(torch.int).reshape((-1, 1))
        # data.x = torch.ones(data.num_nodes).to(torch.int).reshape((-1, 1))
        nx_graph_list[i] = data

    return data_objects# loader

class RoadDataset(InMemoryDataset):
    def __init__(self, root, stage="train", transform=None, pre_transform=None, pre_filter=None, num = 2000):
        self.num = num
        self.stage = stage
        self.stage_to_index = {"train":0,
                               "val":1,
                               "test":2}
        _ = download_roads()
        del _
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
            print("Road files exist")
            return

        data_list = get_road_dataset(num=self.num, targets=self.stage != "train")

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

        del data_list




if __name__ == "__main__":
    # fb_graph = download_cora()
    # print(fb_graph.nodes(data=True))
    # graphs = ESWR(fb_graph, 200, 100)
    # G = download_cora()
    # print(G)
    os.chdir('../')
    dataset = FacebookDataset(os.getcwd()+'/original_datasets/'+'facebook_large', stage = "val")