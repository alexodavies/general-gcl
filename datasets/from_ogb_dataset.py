import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()


# import osmnx as ox
# from ToyDatasets import *


def vis_from_pyg(data, filename = None):
    edges = data.edge_index.T.cpu().numpy()
    labels = data.x[:,0].cpu().numpy().astype(int)

    g = nx.Graph()
    g.add_edges_from(edges)

    fig, ax = plt.subplots(figsize = (6,6))

    pos = nx.kamada_kawai_layout(g)

    nx.draw_networkx_edges(g, pos = pos, ax = ax)
    try:
        if labels.shape[0] > g.order():
            ax.set_title("Labels may be incorrect")
            labels = labels[:g.order()]
        nx.draw_networkx_nodes(g, pos = pos, node_color=labels, cmap="tab20",
                            vmin = 0, vmax = 20, ax = ax)
    except:
        print(labels, labels.shape, edges, np.unique(edges), g)
        quit()

    ax.axis('off')

    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def to_onehot_atoms(x):
    one_hot_tensors = []
    for i, num_values in enumerate(full_atom_feature_dims):
        one_hot = torch.nn.functional.one_hot(x[:, i], num_classes=num_values)
        one_hot_tensors.append(one_hot)

    return torch.cat(one_hot_tensors, dim=1)

def to_onehot_bonds(x):
    one_hot_tensors = []
    for i, num_values in enumerate(full_bond_feature_dims):
        one_hot = torch.nn.functional.one_hot(x[:, i], num_classes=num_values)
        one_hot_tensors.append(one_hot)

    return torch.cat(one_hot_tensors, dim=1)

class FromOGBDataset(InMemoryDataset):
    def __init__(self, root, ogb_dataset, stage = "train", num = -1, transform=None, pre_transform=None, pre_filter=None):
        self.ogb_dataset = ogb_dataset
        self.stage = stage
        self.stage_to_index = {"train":0,
                               "val":1,
                               "test":2}
        self.num = num
        print(f"Converting OGB stage {self.stage}")
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.stage_to_index[self.stage]])

    @property
    def raw_file_names(self):
        return ['dummy.csv']

    @property
    def processed_file_names(self):
        return ['train.pt',
                'val.pt',
                'test.pt']


    def process(self):
        # Read data into huge `Data` list.
        # print(f"Looking for OGB processed files at {self.processed_paths[self.stage_to_index[self.stage]]}")
        if os.path.isfile(self.processed_paths[self.stage_to_index[self.stage]]):
            print(f"OGB files exist at {self.processed_paths[self.stage_to_index[self.stage]]}")
            return
        data_list = self.ogb_dataset# get_fb_dataset(num=self.num)

        num_samples = len(data_list)
        if num_samples < self.num:
            keep_n = num_samples
        else:
            keep_n = self.num

        if self.stage == "train":
            print("Found stage train, dropping targets")
            new_data_list = []
            for i, item in enumerate(data_list[:keep_n]):
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
            for i, item in enumerate(data_list[:keep_n]):
                n_nodes, n_edges = item.x.shape[0], item.edge_index.shape[1]


                data = Data(x = to_onehot_atoms(item.x), # [:,0].reshape((-1, 1)),# torch.ones(n_nodes).to(torch.int).reshape((-1, 1)), #
                            edge_index=item.edge_index,
                            edge_attr= to_onehot_bonds(item.edge_attr), #torch.ones(n_edges).to(torch.int).reshape((-1,1)),
                            y = item.y)

                # data = Data(x = item.x[:,0].reshape((-1, 1)), edge_index=item.edge_index,
                #             edge_attr=item.edge_attr, y = item.y)
                # print(f"Val x shape {data.x.shape}, edge index {data.edge_index.shape}")
                # print(data)
                # vis_from_pyg(data, filename=self.root + '/processed/' + i + '.png')
                new_data_list.append(data)
            data_list = new_data_list
            # for i, data in enumerate(tqdm(data_list)):
            #     vis_from_pyg(data, filename=self.root + f'/processed/{self.stage}-{i}.png')

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
    dataset = FacebookDataset(os.getcwd()+'/original_datasets/'+'facebook_large')