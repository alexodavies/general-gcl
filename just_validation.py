import argparse
import concurrent.futures
import glob
import logging
import os
import random
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import yaml
from bokeh.models import HoverTool, BoxZoomTool, ResetTool, Range1d, UndoTool, WheelZoomTool, PanTool
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from hdbscan import HDBSCAN
from ogb.graphproppred import PygGraphPropPredDataset

from umap import UMAP
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import SpectralEmbedding, Isomap

from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm
from utils import better_to_nx, setup_wandb
from datasets.loaders import get_train_loader, get_val_loaders, get_test_loaders

import wandb
from datasets.community_dataset import CommunityDataset
from datasets.cora_dataset import CoraDataset
from datasets.ego_dataset import EgoDataset
from datasets.facebook_dataset import FacebookDataset
from datasets.from_ogb_dataset import FromOGBDataset
from datasets.neural_dataset import NeuralDataset
from datasets.random_dataset import RandomDataset
from datasets.road_dataset import RoadDataset
from datasets.tree_dataset import TreeDataset
from datasets.lattice_dataset import LatticeDataset



from unsupervised.embedding_evaluation import GeneralEmbeddingEvaluation
from unsupervised.encoder import MoleculeEncoder
from unsupervised.learning import GInfoMinMax
from unsupervised.utils import initialize_edge_weight
from unsupervised.view_learner import ViewLearner


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


class ComponentSlicer:
    def __init__(self, comp_1 = 0, comp_2 = 1):
        self.comp_1 = comp_1
        self.comp_2 = comp_2

    def fit(self, X):
        pass

    def transform(self, X):
        return np.concatenate((X[:, self.comp_1].reshape(-1,1), X[:, self.comp_2].reshape(-1,1)), axis=1)


def average_degree(g):
    return nx.number_of_edges(g)/nx.number_of_nodes(g)
#
def safe_diameter(g):
    """
    Returns either the diameter of a graph or -1 if it has multiple components
    Args:
        g: networkx.Graph

    Returns:
        either the diameter of the graph or -1
    """
    try:
        return nx.diameter(g)
    except:
        return -1

def four_cycles(g):
    """
    Returns the number of 4-cycles in a graph, normalised by the number of nodes
    """
    cycles = nx.simple_cycles(g, 4)
    return len(list(cycles)) / g.order()

def five_cycle_worker(g):
    """
    Returns the number of 5-cycles in a graph, normalised by the number of nodes
    """
    cycles = nx.simple_cycles(g, 5)
    return len(list(cycles)) / nx.number_of_nodes(g)

def five_cycles(graphs):
    """
    Returns the number of 5-cycles per graph in a list of graphs
    Args:
        graphs: list of networkx.Graph objects

    Returns:
        list of 5-cycle counts
    """
    sample_ref = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        for n_coms in tqdm(executor.map(five_cycle_worker, graphs), desc="Five cycles"):
            sample_ref.append(n_coms)

    return sample_ref

def embeddings_vs_metrics(loaders, embeddings, n_components = 10):
    """
    Compute the correlations between a set of metrics and the SVD decomposition of an embedding
    The results are currently printed in-terminal

    Args:
        loaders: set of dataloaders, used to reconstruct networkx.Graph s
        embeddings: tensor or numpy array of embeddings from an encoder
        n_components: the number of components against which to compute correlations

    Returns:
        None (currently)
    """

    # Iterate over items in loaders and convert to nx graphs, while removing selfloops
    val_data = []
    for loader in loaders:
        for batch in loader:
            val_data += batch.to_data_list()

    val_data = [better_to_nx(data)[0] for data in val_data]
    for item in val_data:
        item.remove_edges_from(nx.selfloop_edges(item))


    # Project embedding using SVD
    embedder = TruncatedSVD(n_components=n_components).fit(embeddings)
    projection = embedder.transform(embeddings)

    # Metrics can be added here - should take an nx graph as input and return a numerical value
    metrics = [nx.number_of_nodes, nx.number_of_edges, nx.density, safe_diameter,
               average_degree, nx.average_clustering, nx.transitivity, four_cycles]

    # Compute metrics for all graphs
    metric_arrays = [np.array([metric(g) for g in tqdm(val_data)]) for metric in metrics]

    # 5-cycles is multi-processed, so needs to be handled differently
    metric_arrays += [np.array(five_cycles(val_data))]
    metrics += ["five-cycles"]

    # A little spaghetti but not too expensive compared to computing cycles
    # Iterate over the components of the embedding, then iterate over the metric values
    # to find correlations. The future version will probably produce a .csv
    for component in range(n_components):
        projected = projection[:, component].flatten()

        correlations = [np.corrcoef(projected[metric != -1], metric[metric != -1])[0,1] for metric in metric_arrays]
        max_ind = np.argsort(np.abs(correlations))[-1]

        exp_variance = str(100*embedder.explained_variance_ratio_[component])[:5]
        exp_corr = str(correlations[max_ind])[:5]
        try:
            metric_name = str(metrics[max_ind]).split(' ')[1]
        except:
            metric_name = str(metrics[max_ind])

        print("-"*40)
        print(f"PCA component {component}, explained variance {exp_variance}%, correlated best with {metric_name} (R2: {exp_corr})")
        print("Extra detail:")
        for i_metric, corr in enumerate(correlations):
            metric = metrics[i_metric]
            try:
                metric_name = str(metric).split(' ')[1]
            except:
                metric_name = str(metric)
            print(f"{metric_name} corr. {corr}")

def embeddings_hdbscan(loaders, embeddings, cluster_samples = 100):
    """
    Unfinished function to track how clusters form in the embedding space
    Code is pretty identical to the function above
    Args:
        loaders: a set of validation loaders, used to retrieve networkx graphs
        embeddings: torch tensor or numpy array of encoder embeddings
        cluster_samples: both min_cluster_size and min_samples for HDBSCAN

    Returns:

    """
    val_data = []
    for loader in loaders:
        for batch in loader:
            val_data += batch.to_data_list()

    val_data = [better_to_nx(data)[0] for data in val_data]
    for item in val_data:
        item.remove_edges_from(nx.selfloop_edges(item))

    metrics = [nx.number_of_nodes, nx.number_of_edges, nx.density, safe_diameter,
               average_degree, nx.average_clustering, nx.transitivity]

    metric_arrays = [np.array([metric(g) for g in tqdm(val_data)]) for metric in metrics]

    clusterer = HDBSCAN(min_cluster_size=cluster_samples, min_samples=cluster_samples).fit(embeddings)

    predicted = clusterer.labels_
    unique_clusters, cluster_counts = np.unique(predicted, return_counts=True)

    for i_cluster in range(unique_clusters.shape[0]):
        print(f"---------------\nCluster {unique_clusters[i_cluster]}, size {cluster_counts[i_cluster]}")
        indices = np.arange(predicted.shape[0])[predicted == unique_clusters[i_cluster]]

        specific_metrics = [metric[indices] for metric in metric_arrays]

        for i, metric in enumerate(specific_metrics):
            print(f"{str(metrics[i])}, mean {np.mean(metric)}, std dev {np.std(metric)}")


def wandb_cfg_to_actual_cfg(original_cfg, wandb_cfg):
    """
    Retrive wandb config from saved file
    Args:
        original_cfg: the config from this run
        wandb_cfg: the saved config from the training run

    Returns:
        a config with values updated to those from the saved training run
    """
    original_keys = list(vars(original_cfg).keys())
    wandb_keys = list(wandb_cfg.keys())

    for key in original_keys:
        if key not in wandb_keys:
            continue

        vars(original_cfg)[key] = wandb_cfg[key]['value']

    return original_cfg

def run(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info(args)
    # setup_seed(args.seed)

    num = args.num
    checkpoint = args.checkpoint

    wandb.log({"Linear Transfer":True})
    node_features = args.node_features

    # if checkpoint == "latest":
    #     checkpoint_root = "wandb/latest-run/files"
    #     checkpoints = glob.glob(f"{checkpoint_root}/Checkpoint-*.pt")
    #     print(checkpoints)
    #     epochs = np.array([cp.split('-')[0] for cp in checkpoints])
    #     checkpoint_ind = np.argsort(epochs[::-1])[0]
    #
    #     checkpoint_path = f"wandb/latest-run/files/{checkpoints[checkpoint_ind]}"
    #
    # elif checkpoint == "untrained":
    checkpoint_path = f"outputs/{checkpoint}"
    cfg_name = checkpoint.split('.')[0] + ".yaml"
    config_path = f"outputs/{cfg_name}"

    with open(config_path, 'r') as stream:
        try:
            # Converts yaml document to python object
            wandb_cfg = yaml.safe_load(stream)

            # Printing dictionary
            print(wandb_cfg)
        except yaml.YAMLError as e:
            print(e)

    args = wandb_cfg_to_actual_cfg(args, wandb_cfg)
    #
    # else:
    #     checkpoint_path = f"outputs/{checkpoint}"
    #     cfg_name = checkpoint.split('.')[0] + ".yaml"
    #     config_path = f"outputs/{cfg_name}"
    #
    #     with open(config_path, 'r') as stream:
    #         try:
    #             # Converts yaml document to python object
    #             wandb_cfg = yaml.safe_load(stream)
    #
    #             # Printing dictionary
    #             print(wandb_cfg)
    #         except yaml.YAMLError as e:
    #             print(e)
    #
    #     args = wandb_cfg_to_actual_cfg(args, wandb_cfg)

    model_name = checkpoint_path.split("/")[-1].split(".")[0]
    wandb.log({"Model Name": model_name})

    # Retrieved saved models and load weights

    model = GInfoMinMax(MoleculeEncoder(emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
                    proj_hidden_dim=args.emb_dim).to(device)

    view_learner = ViewLearner(MoleculeEncoder(emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
                               mlp_edge_model_dim=args.mlp_edge_model_dim).to(device)

    if checkpoint != "untrained":
        model_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_dict['encoder_state_dict'])
        view_learner.load_state_dict(model_dict['view_state_dict'])

    # Get datasets
    my_transforms = Compose([initialize_edge_weight])

    val_loaders, names = get_test_loaders(args.batch_size, my_transforms, num=num)
    train_loaders, names = get_val_loaders(args.batch_size, my_transforms, num=2*num)

    # Get embeddings
    general_ee = GeneralEmbeddingEvaluation()
    model.eval()
    # all_embeddings, separate_embeddings = general_ee.get_embeddings(model.encoder, val_loaders)

    general_ee.embedding_evaluation(model.encoder, train_loaders, val_loaders, names,
                                    node_features=node_features, not_in_training=True)




def arg_parse():
    parser = argparse.ArgumentParser(description='AD-GCL ogbg-mol*')

    parser.add_argument('--dataset', type=str, default='ogbg-molesol',
                        help='Dataset')
    parser.add_argument('--model_lr', type=float, default=0.001,
                        help='Model Learning rate.')
    parser.add_argument('--view_lr', type=float, default=0.001,
                        help='View Learning rate.')
    parser.add_argument('--num_gc_layers', type=int, default=5,
                        help='Number of GNN layers before pooling')
    parser.add_argument('--pooling_type', type=str, default='standard',
                        help='GNN Pooling Type Standard/Layerwise')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimension')
    parser.add_argument('--mlp_edge_model_dim', type=int, default=128,
                        help='embedding dimension')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.2,
                        help='Dropout Ratio / Probability')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='Train Epochs')
    parser.add_argument('--reg_lambda', type=float, default=5.0, help='View Learner Edge Perturb Regularization Strength')

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--num', type=int, default=5000,
                        help='Number of points included in each dataset')

    parser.add_argument('--redo_views', type=bool, default=False,
                        help='Whether to re-vis views')

    parser.add_argument('--checkpoint', type=str, default="latest", help='Either the name of the trained model checkpoint in ./outputs/, or latest for the most recent trained model in ./wandb/latest-run/files')

    parser.add_argument(
        '-f',
        '--node-features',
        action='store_true',
        help='Whether to include node features (labels) in evaluation',
    )

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()
    setup_wandb(args)
    run(args)

