import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def format_features(features, key_list):
    """Helper function to format the features with their corresponding keys."""
    return ", ".join([f"{key}: {value}" for key, value in zip(key_list, features)])

def edge_list_to_text_with_multivariate_features(edge_list, node_features, key_list):
    if isinstance(edge_list, list):
        edge_list = np.array(edge_list)

    if edge_list.shape[1] != 2:
        edge_list = edge_list.T
    assert edge_list.shape[1] == 2, "Each edge should have 2 nodes."
    # Ensure the number of features matches the number of keys
    
    connections = []
    node_info = []
    
    # Part 1: Connections
    for edge in edge_list:
        node1, node2 = edge
        connections.append(f"Node {node1} is connected to node {node2}.")
    
    if len(node_features) == 0:
        return " ".join(connections)
    # Part 2: Node features
    feature_dim = len(key_list)
    for node_id, features in enumerate(node_features):
        if len(features) != feature_dim:
            raise ValueError(f"Mismatch between the number of features and keys for node {node_id}.")
        
        feature_str = format_features(features, key_list)
        node_info.append(f"Node {node_id}: {feature_str}")
    
    # Join connections and node info
    return " ".join(connections), "\n".join(node_info)



if __name__ == "__main__":
    # Example usage
    # open pickle file
    with open('simulation_data.json', 'rb') as f:
        data = pkl.load(f)
        
    adj_mat = np.load('network_adj_mat.npy')

    all_keys = list(data[0].keys())

    selected_keys = all_keys[-4:]

    temporal_graph_data = {}
    for selected_key in selected_keys:
        node_values_per_timestep = []
        for i in range(len(data)):
            node_values_per_timestep.append(data[i][selected_key])
            
        node_values_per_timestep = np.array(node_values_per_timestep)
        
        temporal_graph_data[selected_key] = node_values_per_timestep
        
        
    # Load node features (time series data)
    node_features_list = np.array([temporal_graph_data[key].T for key in selected_keys]).T # Shape: (T, N, d)
    # Load adjacency matrix
    adj_matrix = adj_mat  # Shape: (C, C)

    # Convert adjacency matrix to edge list format for PyTorch Geometric
    edge_index = torch_geometric.utils.dense_to_sparse(torch.tensor(adj_matrix, dtype=torch.float32))[0]
    edge_index_list = [edge_index for _ in range(len(node_features_list))]
    node_features_list = torch.tensor(node_features_list, dtype=torch.float32)
    edge_index_list = torch.stack(edge_index_list) # Shape: (T, 2, E)
    print(node_features_list.shape)

    print(selected_key)

    print(edge_index_list.shape) 

    edge_list = edge_index_list[0]
    node_features = node_features_list[0]
    key_list = selected_keys
    connections, node_info = edge_list_to_text_with_multivariate_features(edge_list, node_features, key_list)
    print(connections)
    print(node_info)    