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
    if stage == "train":
        names = ["ogbg-molpcba"]
    else:
        names = ["ogbg-molesol","ogbg-molclintox", "ogbg-molfreesolv", "ogbg-mollipo"]

    datasets = [PygGraphPropPredDataset(name=name, root='./original_datasets/', transform=transforms) for name in names]

    split_idx = [data.get_idx_split() for data in datasets]

    if stage == "val":
        datasets = [data[split_idx[i]["valid"]] for i, data in enumerate(datasets)]
    else:
        datasets = [data[split_idx[i][stage]] for i, data in enumerate(datasets)]

    # Need to convert to pyg inmemorydataset
    datasets = [FromOGBDataset(os.getcwd()+'/original_datasets/'+names[i], data, num=num, stage=stage) for i, data in enumerate(datasets)]

    return datasets, names

def get_social_datasets(transforms, num, stage = "train"):
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

def get_train_loader(batch_size, transforms, num_social = 20000):
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



    # # Open graph benchmark datasets
    # names = ["ogbg-molpcba"]
    # datasets = [PygGraphPropPredDataset(name=name, root='./original_datasets/', transform=transforms) for name in names]
    #
    # split_idx = [data.get_idx_split() for data in datasets]
    # datasets = [data[split_idx[i]["train"]] for i, data in enumerate(datasets)]
    # datasets = [FromOGBDataset(os.getcwd() + '/original_datasets/' + names[i], data, num=num_social) for i, data in
    #             enumerate(datasets)]

    datasets, _ = get_chemical_datasets(transforms, num_social, stage="train")
    social_datasets, _ = get_social_datasets(transforms, num_social, stage="train")

    datasets += social_datasets

    # Need to convert to pyg inmemorydataset
    combined = []
    # combined = FromOGBDataset(os.getcwd()+'/original_datasets/'+'ogbg-molesol', dataset, num=num_social)
    # Concat dataset of ogb
    for data in datasets:
        combined += data
    # for data in social_datasets:
    #     combined += data

    # Get other datasets (see /datasets/*)
    # social_datasets = [transforms(FacebookDataset(os.getcwd()+'/original_datasets/'+'facebook_large', num=num_social)),
    #                    transforms(EgoDataset(os.getcwd()+'/original_datasets/'+'twitch_egos', num=num_social, stage="train")),
    #                    transforms(CoraDataset(os.getcwd()+'/original_datasets/'+'cora', num=num_social)),
    #                    transforms(RoadDataset(os.getcwd() + '/original_datasets/' + 'roads', stage="train", num=num_social)),
    #                    transforms(NeuralDataset(os.getcwd()+'/original_datasets/'+'fruit_fly', stage = "train", num=num_social))]

    # Final concat dataset


    return DataLoader(combined, batch_size=batch_size, shuffle=True)

def get_val_loaders(batch_size, transforms, num = 2000):
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

    chemical_datasets, ogbg_names = get_chemical_datasets(transforms, num, stage="val")
    social_datasets, social_names = get_social_datasets(transforms, num, stage="val")

    datasets = chemical_datasets + social_datasets

    # ogbg_names = ["ogbg-molesol","ogbg-molclintox", "ogbg-molfreesolv", "ogbg-mollipo"]
    #
    # social_datasets = [transforms(FacebookDataset(os.getcwd() +'/original_datasets/' +'facebook_large', stage = "val", num=num)),
    #                    transforms(EgoDataset(os.getcwd() +'/original_datasets/' +'twitch_egos', stage = "val", num=num)),
    #                    transforms(CoraDataset(os.getcwd() +'/original_datasets/' +'cora', stage = "val", num=num)),
    #                    transforms(RoadDataset(os.getcwd() + '/original_datasets/' + 'roads', stage="val", num=num)),
    #                    transforms(NeuralDataset(os.getcwd() +'/original_datasets/' +'fruit_fly', stage = "val", num=num)),
    #                    transforms(TreeDataset(os.getcwd() +'/original_datasets/' +'trees', stage = "val", num=num)),
    #                    transforms(RandomDataset(os.getcwd() + '/original_datasets/' + 'random', stage="val", num=num)),
    #                    transforms(CommunityDataset(os.getcwd() +'/original_datasets/' +'community', stage = "val", num=num))
    #                    ]
    #
    # # For each open graph benchmark dataset, move back to a pyg.data.InMemoryDataset
    # datasets = [PygGraphPropPredDataset(name=name, root='./original_datasets/', transform=transforms) for name in ogbg_names]
    # split_idx = [data.get_idx_split() for data in datasets]
    #
    # # Get validation splits for each ogbg dataset, and trim if longer than num
    # datasets = [data[split_idx[i]["test"]] for i, data in enumerate(datasets)]
    # dataset_lengths = [len(data) for data in datasets]
    #
    # for i, data in enumerate(datasets):
    #     if dataset_lengths[i] > num:
    #         datasets[i] = data[:num]
    # print("\n", datasets, "\n")
    # datasets = [FromOGBDataset(os.getcwd() +'/original_datasets/' + ogbg_names[i], data, stage = "val", num=num) for i, data in enumerate(datasets)]

    # datasets = datasets
    # all_datasets = datasets + social_datasets

    datasets = [DataLoader(data, batch_size=batch_size) for data in datasets]

    return datasets, ogbg_names + social_names

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

    chemical_datasets, ogbg_names = get_chemical_datasets(transforms, num, stage="test")
    social_datasets, social_names = get_social_datasets(transforms, num, stage="test")

    datasets = chemical_datasets + social_datasets

    # ogbg_names = ["ogbg-molesol","ogbg-molclintox", "ogbg-molfreesolv", "ogbg-mollipo"]
    #
    # social_datasets = [transforms(FacebookDataset(os.getcwd() +'/original_datasets/' +'facebook_large', stage = "val", num=num)),
    #                    transforms(EgoDataset(os.getcwd() +'/original_datasets/' +'twitch_egos', stage = "val", num=num)),
    #                    transforms(CoraDataset(os.getcwd() +'/original_datasets/' +'cora', stage = "val", num=num)),
    #                    transforms(RoadDataset(os.getcwd() + '/original_datasets/' + 'roads', stage="val", num=num)),
    #                    transforms(NeuralDataset(os.getcwd() +'/original_datasets/' +'fruit_fly', stage = "val", num=num)),
    #                    transforms(TreeDataset(os.getcwd() +'/original_datasets/' +'trees', stage = "val", num=num)),
    #                    transforms(RandomDataset(os.getcwd() + '/original_datasets/' + 'random', stage="val", num=num)),
    #                    transforms(CommunityDataset(os.getcwd() +'/original_datasets/' +'community', stage = "val", num=num))
    #                    ]
    #
    # # For each open graph benchmark dataset, move back to a pyg.data.InMemoryDataset
    # datasets = [PygGraphPropPredDataset(name=name, root='./original_datasets/', transform=transforms) for name in ogbg_names]
    # split_idx = [data.get_idx_split() for data in datasets]
    #
    # # Get validation splits for each ogbg dataset, and trim if longer than num
    # datasets = [data[split_idx[i]["test"]] for i, data in enumerate(datasets)]
    # dataset_lengths = [len(data) for data in datasets]
    #
    # for i, data in enumerate(datasets):
    #     if dataset_lengths[i] > num:
    #         datasets[i] = data[:num]
    # print("\n", datasets, "\n")
    # datasets = [FromOGBDataset(os.getcwd() +'/original_datasets/' + ogbg_names[i], data, stage = "val", num=num) for i, data in enumerate(datasets)]

    # datasets = datasets
    # all_datasets = datasets + social_datasets

    datasets = [DataLoader(data, batch_size=batch_size) for data in datasets]

    return datasets, ogbg_names + social_names

# def get_test_loaders(batch_size, transforms, num = 1000):
#     """
#     Get a list of validation loaders
#
#     Args:
#         dataset: the -starting dataset-, a hangover from previous code, likely to be gone in the next refactor
#         batch_size: batch size for loaders
#         transforms: a set of transforms applied to the data
#         num: the maximum number of samples in each dataset (and therefore dataloader)
#
#     Returns:
#         datasets: list of dataloaders
#         names: name of each loaders respective dataset
#
#     """
#     ogbg_names = ["ogbg-molesol", "ogbg-molclintox", "ogbg-molfreesolv", "ogbg-mollipo"]
#
#     social_datasets = [transforms(FacebookDataset(os.getcwd() +'/original_datasets/' +'facebook_large', stage = "test", num=num)),
#                        transforms(EgoDataset(os.getcwd() +'/original_datasets/' +'twitch_egos', stage = "test", num=num)),
#                        transforms(CoraDataset(os.getcwd() +'/original_datasets/' +'cora', stage = "test", num=num)),
#                        transforms(RoadDataset(os.getcwd() + '/original_datasets/' + 'roads', stage="test", num=num)),
#                        transforms(NeuralDataset(os.getcwd() +'/original_datasets/' +'fruit_fly', stage = "test", num=num)),
#                        transforms(TreeDataset(os.getcwd() +'/original_datasets/' +'trees', stage = "test", num=num)),
#                        transforms(RandomDataset(os.getcwd() + '/original_datasets/' + 'random', stage="test", num=num)),
#                        transforms(CommunityDataset(os.getcwd() +'/original_datasets/' +'community', stage = "test", num=num))
#                        ]
#
#     # For each open graph benchmark dataset, move back to a pyg.data.InMemoryDataset
#     datasets = [PygGraphPropPredDataset(name=name, root='./original_datasets/', transform=transforms) for name in ogbg_names]
#     split_idx = [data.get_idx_split() for data in datasets]
#
#     # Get validation splits for each ogbg dataset, and trim if longer than num
#     datasets = [data[split_idx[i]["test"]] for i, data in enumerate(datasets)]
#     dataset_lengths = [len(data) for data in datasets]
#
#     for i, data in enumerate(datasets):
#         if dataset_lengths[i] > num:
#             datasets[i] = data[:num]
#     print("\n", datasets, "\n")
#     datasets = [FromOGBDataset(os.getcwd() +'/original_datasets/' + ogbg_names[i], data, stage = "test", num=num) for i, data in enumerate(datasets)]
#
#     # datasets = datasets
#     all_datasets = datasets + social_datasets
#
#     datasets = [DataLoader(data, batch_size=batch_size) for data in all_datasets]
#
#     return datasets, ogbg_names + ["facebook_large", "twitch_egos", "cora", "roads", "fruit_fly",
#                                    "trees", "random", "community"]