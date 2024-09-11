import torch
from tqdm import tqdm
from .feature_noise import add_continuous_feature_noise, add_discrete_feature_noise, shuffle_categorical_feature_noise, weighted_categorical_feature_noise, dataset_shuffle_feature_noise
from .structure_noise import add_structure_noise

def add_noise_to_graph(data, t_structure, t_feature):
    data = add_structure_noise(data, t_structure)
    # data = add_discrete_feature_noise(data, t_feature)
    # data = shuffle_categorical_feature_noise(data, t_feature)
    # data = add_continuous_feature_noise(data, t_feature)
    return data

def add_noise_to_dataset(dataset, t_structure, t_feature):
    noisy_dataset = []
    
    for data in tqdm(dataset, desc='Adding noise', colour='green', leave=False):
        noisy_data = add_noise_to_graph(data, t_structure, t_feature)
        noisy_dataset.append(noisy_data)

    noisy_dataset = dataset_shuffle_feature_noise(noisy_dataset, t_feature)
    
    return noisy_dataset

def add_weighted_noise_to_graph(data, t_structure, t_feature, weights_nodes, weights_edges):
    data = add_structure_noise(data, t_structure)
    # data = add_discrete_feature_noise(data, t_feature)
    data = weighted_categorical_feature_noise(data, t_feature, weights_nodes, weights_edges)
    return data

def add_weighted_noise_to_dataset(dataset, t_structure, t_feature, weights_nodes, weights_edges):
    noisy_dataset = []
    
    for data in tqdm(dataset, desc='Adding noise', colour='green', leave=False):
        noisy_data = add_noise_to_graph(data, t_structure, t_feature)
        noisy_dataset.append(noisy_data)
    
    return noisy_dataset

def compute_onehot_probabilities(dataloader):
    n_feats = None
    total = None
    n_points = 0

    for data in dataloader:
        if n_feats is None:
            n_feats = data.x.shape[1]
            total = torch.zeros(n_feats)
        
        n_points += data.x.shape[0]
        total += data.x.sum(dim=0)

    return total / n_points

def compute_onehot_probabilities_edge(dataloader):
    n_feats = None
    total = None
    n_points = 0

    for data in dataloader:
        if n_feats is None:
            n_feats = data.edge_attr.shape[1]
            total = torch.zeros(n_feats)
        
        n_points += data.edge_attr.shape[0]
        total += data.edge_attr.sum(dim=0)

    return total / n_points

