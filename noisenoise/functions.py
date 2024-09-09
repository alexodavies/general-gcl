import torch
from tqdm import tqdm
from .feature_noise import add_continuous_feature_noise, add_discrete_feature_noise
from .structure_noise import add_structure_noise

def add_noise_to_graph(data, t_structure, t_feature):
    data = add_structure_noise(data, t_structure)
    data = add_discrete_feature_noise(data, t_feature)

    return data

def add_noise_to_dataset(dataset, t_structure, t_feature):
    noisy_dataset = []
    
    for data in tqdm(dataset, desc='Adding noise', colour='green', leave=False):
        noisy_data = add_noise_to_graph(data, t_structure, t_feature)
        noisy_dataset.append(noisy_data)
    
    return noisy_dataset
