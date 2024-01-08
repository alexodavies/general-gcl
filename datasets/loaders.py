import os
from torch_geometric.data import DataLoader

from ogb.graphproppred import PygGraphPropPredDataset
from datasets.facebook_dataset import FacebookDataset
from datasets.ego_dataset import EgoDataset
from datasets.community_dataset import CommunityDataset
from datasets.cora_dataset import CoraDataset
from datasets.random_dataset import RandomDataset
from datasets.neural_dataset import NeuralDataset
from datasets.road_dataset import RoadDataset
from datasets.tree_dataset import TreeDataset
from datasets.lattice_dataset import LatticeDataset
from datasets.from_ogb_dataset import FromOGBDataset


def get_chemical_datasets(transforms, num, stage="train"):
    if "original_datasets" not in os.listdir():
        os.mkdir("original_datasets")

    if stage == "train":
        names = ["ogbg-molpcba"]
    else:
        names = ["ogbg-molpcba", "ogbg-molesol","ogbg-molclintox",
                 "ogbg-molfreesolv", "ogbg-mollipo", "ogbg-molhiv",
                "ogbg-molbbbp", "ogbg-molbace",
                 ]

    datasets = [PygGraphPropPredDataset(name=name, root='./original_datasets/', transform=transforms) for name in names]

    split_idx = [data.get_idx_split() for data in datasets]

    if stage == "val":
        datasets = [data[split_idx[i]["valid"]] for i, data in enumerate(datasets)]
    else:
        datasets = [data[split_idx[i][stage]] for i, data in enumerate(datasets)]

    # Need to convert to pyg inmemorydataset
    num = num if stage != "train" else 5*num
    datasets = [FromOGBDataset(os.getcwd()+'/original_datasets/'+names[i],
                               data,
                               num=num, stage = stage) for i, data in enumerate(datasets)] #  if names[i] != "ogbg-molpcba" else 5*num, stage=stage

    return datasets, names

def get_social_datasets(transforms, num, stage = "train"):
    if "original_datasets" not in os.listdir():
        os.mkdir("original_datasets")

    if stage == "train":
        social_datasets = [
            transforms(FacebookDataset(os.getcwd() + '/original_datasets/' + 'facebook_large', num=num)),
            transforms(EgoDataset(os.getcwd() + '/original_datasets/' + 'twitch_egos', num=num, stage=stage)),
            transforms(CoraDataset(os.getcwd() + '/original_datasets/' + 'cora', num=num)),
            transforms(RoadDataset(os.getcwd() + '/original_datasets/' + 'roads', stage=stage, num=num)),
            transforms(NeuralDataset(os.getcwd() + '/original_datasets/' + 'fruit_fly', stage=stage, num=num))]
        names = ["facebook_large", "twitch_egos", "cora", "roads", "fruit_fly"]
    elif stage == "val":
        social_datasets = [
            transforms(FacebookDataset(os.getcwd() + '/original_datasets/' + 'facebook_large', stage=stage, num=num)),
            transforms(EgoDataset(os.getcwd() + '/original_datasets/' + 'twitch_egos', stage=stage, num=num)),
            transforms(CoraDataset(os.getcwd() + '/original_datasets/' + 'cora', stage=stage, num=num)),
            transforms(RoadDataset(os.getcwd() + '/original_datasets/' + 'roads', stage=stage, num=num)),
            transforms(NeuralDataset(os.getcwd() + '/original_datasets/' + 'fruit_fly', stage=stage, num=num)),
            transforms(TreeDataset(os.getcwd() + '/original_datasets/' + 'trees', stage=stage, num=num)),
            transforms(RandomDataset(os.getcwd() + '/original_datasets/' + 'random', stage=stage, num=num)),
            transforms(CommunityDataset(os.getcwd() + '/original_datasets/' + 'community', stage=stage, num=num))
            ]
        names = ["facebook_large", "twitch_egos", "cora", "roads", "fruit_fly", "trees", "random", "community"]
    else:
        social_datasets = [
            transforms(FacebookDataset(os.getcwd() + '/original_datasets/' + 'facebook_large', stage=stage, num=num)),
            transforms(EgoDataset(os.getcwd() + '/original_datasets/' + 'twitch_egos', stage=stage, num=num)),
            transforms(CoraDataset(os.getcwd() + '/original_datasets/' + 'cora', stage=stage, num=num)),
            transforms(RoadDataset(os.getcwd() + '/original_datasets/' + 'roads', stage=stage, num=num)),
            transforms(NeuralDataset(os.getcwd() + '/original_datasets/' + 'fruit_fly', stage=stage, num=num)),
            transforms(TreeDataset(os.getcwd() + '/original_datasets/' + 'trees', stage=stage, num=num)),
            transforms(RandomDataset(os.getcwd() + '/original_datasets/' + 'random', stage=stage, num=num)),
            transforms(CommunityDataset(os.getcwd() + '/original_datasets/' + 'community', stage=stage, num=num))
            ]
        names = ["facebook_large", "twitch_egos", "cora", "roads", "fruit_fly", "trees", "random", "community"]

    return social_datasets, names

def get_train_loader(batch_size, transforms, subset = ["chemical", "social"], num_social = 50000):
    """
    Prepare a torch concat dataset dataloader
    Args:
        dataset: original dataset - hangover from early code, will be removed in future version
        batch_size: batch size for dataloader
        transforms: transforms applied to each dataset
        num_social: number of graphs to sample for each dataset

    Returns:
        dataloader for concat dataset
    """
    if "chemical" in subset:
        datasets, _ = get_chemical_datasets(transforms, num_social, stage="train")
    else:
        print("Skipping chemicals")
        datasets = []

    if "social" in subset:
        social_datasets, _ = get_social_datasets(transforms, num_social, stage="train")
    else:
        print("Skipping socials")
        social_datasets = []

    datasets += social_datasets
    combined = []
    # Concat dataset
    print(datasets)
    for data in datasets:
        combined += data

    return DataLoader(combined, batch_size=batch_size, shuffle=True)

def get_test_loaders(batch_size, transforms, num = 2000):
    """
    Get a list of validation loaders

    Args:
        dataset: the -starting dataset-, a hangover from previous code, likely to be gone in the next refactor
        batch_size: batch size for loaders
        transforms: a set of transforms applied to the data
        num: the maximum number of samples in each dataset (and therefore dataloader)

    Returns:
        datasets: list of dataloaders
        names: name of each loaders' respective dataset

    """

    datasets, names = get_test_datasets(transforms, num=num)
    datasets = [DataLoader(data, batch_size=batch_size) for data in datasets]

    return datasets, names

def get_test_datasets(transforms, num = 2000):

    chemical_datasets, ogbg_names = get_chemical_datasets(transforms, num, stage="test")
    social_datasets, social_names = get_social_datasets(transforms, num, stage="test")

    datasets = chemical_datasets + social_datasets

    return datasets, ogbg_names + social_names

def get_val_loaders(batch_size, transforms, num = 5000):
    """
    Get a list of validation loaders

    Args:
        dataset: the -starting dataset-, a hangover from previous code, likely to be gone in the next refactor
        batch_size: batch size for loaders
        transforms: a set of transforms applied to the data
        num: the maximum number of samples in each dataset (and therefore dataloader)

    Returns:
        datasets: list of dataloaders
        names: name of each loaders' respective dataset

    """

    datasets, names = get_val_datasets(transforms, num = num)
    datasets = [DataLoader(data, batch_size=batch_size) for data in datasets]

    return datasets, names

def get_val_datasets(transforms, num = 2000):

    chemical_datasets, ogbg_names = get_chemical_datasets(transforms, num, stage="val")
    social_datasets, social_names = get_social_datasets(transforms, num, stage="val")

    datasets = chemical_datasets + social_datasets

    return datasets, ogbg_names + social_names

def get_train_datasets(transforms, num = 2000):

    chemical_datasets, ogbg_names = get_chemical_datasets(transforms, num, stage="train")
    social_datasets, social_names = get_social_datasets(transforms, num, stage="train")

    datasets = chemical_datasets + social_datasets

    return datasets, ogbg_names + social_names

