import os
import networkx as nx
import torch
from tqdm import tqdm
import torch_geometric as pyg
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils.convert import to_networkx
from torch_geometric.io import read_npz
# import osmnx as ox
from littleballoffur.exploration_sampling import MetropolisHastingsRandomWalkSampler
# from ToyDatasets import *
import wget
# from utils import vis_from_pyg
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print(os.getcwd())

from utils import ESWR

def five_cycle_worker(g):
    """
    Returns the number of 5-cycles in a graph, normalised by the number of nodes
    """
    cycles = nx.simple_cycles(g, length_bound = 5)
    return len(list(cycles))

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

def download_cora(visualise = False):
    zip_url = "https://github.com/abojchevski/graph2gauss/raw/master/data/cora_ml.npz"

    start_dir = os.getcwd()
    # print(os.getcwd(), os.listdir())
    os.chdir("original_datasets")

    if "cora" not in os.listdir():
        print("Downloading CORA")
        os.mkdir("cora")
        os.chdir("cora")
        _ = wget.download(zip_url)
    else:
        os.chdir("cora")

    edges = read_npz("cora_ml.npz")
    G = to_networkx(edges, to_undirected=True)

    node_classes = {n: int(edges.y[i].item()) for i, n in enumerate(list(G.nodes()))}

    # print(node_classes)

    # base_tensor = torch.Tensor([0])

    for node in list(G.nodes()):
        # class_tensor = base_tensor.clone()
        class_tensor = torch.Tensor([node_classes[node]])#node_classes[node]

        # print(class_tensor, node_classes[node])

        G.nodes[node]["attrs"] = class_tensor

    for edge in list(G.edges()):
        G.edges[edge]["attrs"] = torch.Tensor([1])

    CGs = [G.subgraph(c) for c in nx.connected_components(G)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    graph = CGs[0]
    graph = nx.convert_node_labels_to_integers(graph)

    os.chdir(start_dir)
    return graph


def get_cora_dataset(num = 2000, targets = False):
    fb_graph = download_cora()
    # print(fb_graph.nodes(data=True))
    nx_graph_list = ESWR(fb_graph, num, 48)

    # loader = pyg.loader.DataLoader([pyg.utils.from_networkx(g, group_node_attrs=all, group_edge_attrs=all) for g in nx_graph_list],
    #                                           batch_size=batch_size)

    data_objects = [pyg.utils.from_networkx(g, group_node_attrs=all, group_edge_attrs=all) for g in nx_graph_list]
    for i_data, data in enumerate(tqdm(data_objects, desc="Calculating five cycle values for cora", leave=False)):
        if targets:
            data.y = torch.tensor(five_cycle_worker(nx_graph_list[i_data]) * 0.01) # None # torch.Tensor([[0,0]])
        else:
            data.y = torch.tensor([1])

    return  data_objects# loader

class CoraDataset(InMemoryDataset):
    def __init__(self, root, stage = "train", transform=None, pre_transform=None, pre_filter=None, num = 2000):
        self.num = num
        self.stage = stage
        self.stage_to_index = {"train":0,
                               "val":1,
                               "test":2}
        _ = download_cora()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.stage_to_index[self.stage]])


    @property
    def raw_file_names(self):
        return ['cora_ml.npz']

    @property
    def processed_file_names(self):
        return ['train.pt',
                'val.pt',
                'test.pt']


    def process(self):
        # Read data into huge `Data` list.

        if os.path.isfile(self.processed_paths[self.stage_to_index[self.stage]]):
            print("Cora files exist")
            return

        data_list = get_cora_dataset(num=self.num, targets=self.stage != "train")

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
    o

    dataset = CoraDataset(os.getcwd()+'/original_datasets/'+'cora')