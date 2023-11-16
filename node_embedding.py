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
import pandas as pd
import torch
import yaml
from bokeh.models import HoverTool, BoxZoomTool, ResetTool, Range1d, UndoTool, WheelZoomTool, PanTool
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from hdbscan import HDBSCAN
from ogb.graphproppred import PygGraphPropPredDataset

from umap import UMAP
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import SpectralEmbedding, Isomap

from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

import wandb
from utils import better_to_nx, vis_from_pyg, vis_grid, setup_wandb
from datasets.loaders import get_train_loader, get_val_loaders, get_test_loaders
from seaborn import kdeplot

from sklearn.preprocessing import StandardScaler

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



from unsupervised.embedding_evaluation import GeneralEmbeddingEvaluation, NodeEmbeddingEvaluation
from unsupervised.encoder import Encoder, NodeEncoder
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
    if nx.number_of_edges(g)/nx.number_of_nodes(g) < 1:
        print(g)
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

def three_cycles(g):
    """
    Returns the number of 4-cycles in a graph, normalised by the number of nodes
    """
    cycles = nx.simple_cycles(g, 3)
    return len(list(cycles))

def four_cycles(g):
    """
    Returns the number of 4-cycles in a graph, normalised by the number of nodes
    """
    cycles = nx.simple_cycles(g, 4)
    return len(list(cycles))

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

def tidy_return(g, function = nx.degree):
    # result = list(function(g))
    # if type(result[0]) not in [int, float]:
    #     result = [ires[1] for ires in len(result)]

    result = []
    nx.convert_node_labels_to_integers(g)
    for n in range(g.order()):
        # try:
        print(g.nodes, n)
        result.append(function(g, nbunch = n))
        # except:
        #     result.append(-1000)
    return result

def prettify_metric_name(metric):
    try:
        metric_name = str(metric).split(' ')[1]
    except:
        metric_name = str(metric)
    # metrics = [nx.number_of_nodes, nx.number_of_edges, nx.density, safe_diameter,
    #            average_degree, nx.average_clustering, nx.transitivity]
    # pretty_dict = {"number_of_nodes": "Num. Nodes",
    #                "number_of_edges": "Num. Edges",
    #                "density": "Density",
    #                "safe_diameter": "Diameter",
    #                "average_degree": "Avg. Degree",
    #                "average_clustering": "Avg. Clustering",
    #                "transitivity": "Transitivity",
    #                "three_cycles": "Num. 3-Cycles",
    #                "four_cycles":"Num. 4-Cycles"}
    pretty_dict = {"degree":"Degree",
                   "clustering":"Clustering"}


    return pretty_dict[metric_name]

def clean_graph(g):
    Gcc = sorted(nx.connected_components(g), key=len, reverse=True)
    g = g.subgraph(Gcc[0]).copy()
    g.remove_edges_from(nx.selfloop_edges(g))

    return g

def get_metric_values(loaders):

    # Iterate over items in loaders and convert to nx graphs, while removing selfloops
    val_data = []
    for loader in loaders:
        for batch in loader:
            val_data += batch.to_data_list()

    val_data = [better_to_nx(data)[0] for data in val_data]
    for i, item in enumerate(val_data):
        # item.remove_edges_from(nx.selfloop_edges(item))
        val_data[i] = clean_graph(item)

    # Metrics can be added here - should take an nx graph as input and return a numerical value
    metrics = [nx.degree, nx.clustering] #, average_degree, ]


    metric_names = [prettify_metric_name(metric) for metric in metrics]
    # Compute metrics for all graphs

    metric_arrays = [np.array(sum((tidy_return(g, function=metric) for g in tqdm(val_data, leave=False, desc=metric_names[i_metric])), [])) for i_metric, metric in enumerate(metrics)]

    return metrics, metric_arrays, metric_names

def embeddings_vs_metrics(metrics, metric_arrays, metric_names, embeddings, n_components = 5, model_name = ""):
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

    embedder = PCA().fit(embeddings)
    print(model_name)
    print(f"Explained variances total {np.sum(embedder.explained_variance_ratio_)}")
    projection = embedder.transform(embeddings)

    # 5-cycles is multi-processed, so needs to be handled differently
    # metric_arrays += [np.array(five_cycles(val_data[::10]))]
    # metrics += ["five-cycles"]

    # A little spaghetti but not too expensive compared to computing cycles
    # Iterate over the components of the embedding, then iterate over the metric values
    # to find correlations. The future version will probably produce a .csv
    print_items = ""
    for component in range(n_components):
        projected = projection[:, component].flatten()
        try:
            correlations = [np.corrcoef(projected[metric != -1], metric[metric != -1])[0,1] for metric in metric_arrays]
        except:
            correlations = [np.corrcoef(projected[::10][metric != -1], metric[metric != -1])[0, 1] for metric in
                            metric_arrays]
        print(f"Comp {component}, correlations: {correlations}")
        max_inds = np.argsort(np.abs(correlations))

        # for ind in max_inds[:3].tolist():
        component_vs_metric(metric_arrays[max_inds[-1]], projected, metric = metrics[max_inds[-1]], model_name = model_name + f"-component-{component}")

        if component < 6:
            correlations = [np.around(corr, decimals=3) for corr in correlations]
            exp_variance_string = embedder.explained_variance_ratio_[component]
            exp_variance_string = float('%.3g' % exp_variance_string)
            exp_variance_string = python_index_to_latex('{:.2e}'.format(exp_variance_string))

            print_items += f"PCA {component} & {exp_variance_string} & {metric_names[max_inds[-1]]} & {correlations[max_inds[-1]]} & {metric_names[max_inds[-2]]} & {correlations[max_inds[-2]]} & {metric_names[max_inds[-3]]} & {correlations[max_inds[-3]]} \\\\ \n"

    print(print_items)

def component_vs_metric(metric_values, component, metric = nx.number_of_nodes, model_name = ""):
    # print(f"Corr. value: {np.corrcoef(metric_values, component)}")

    fig, ax = plt.subplots(figsize=(3.25,3))

    outliers_metric, outliers_comp = is_outlier(metric_values, thresh=2), is_outlier(component)
    outliers = outliers_metric + outliers_comp

    x_min, x_max = np.min(metric_values[~outliers]), np.max(metric_values[~outliers])
    y_min, y_max = np.min(component[~outliers]), np.max(component[~outliers])

    # metric_values = metric_values[~outliers]
    # component = component[~outliers]

    # print(metric_values.shape, component.shape)

    ax.scatter(x = metric_values, y = component, marker="x",
               alpha=0.2, s = 1, c = "black")

    ax.set_xlabel(f"{prettify_metric_name(metric)}")
    ax.set_ylabel(f"Component-{model_name[-1]}")

    if x_min == 0:
        x_min -= 0.2*x_max
    elif x_min < 0:
        x_min = 1.3 * x_min
    else:
        x_min = 0.7 * x_min

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

    ax.grid(False)

    plt.tight_layout()
    plt.savefig(f"outputs/{model_name}-{prettify_metric_name(metric)}.png", dpi = 600)
    plt.close()

def is_outlier(points, thresh=3):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def python_index_to_latex(index: str):
    index = index.replace('e', r'\times 10^{')
    index = "$" + index + '}$'
    return index

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
    redo_views = args.redo_views

    # Get datasets
    my_transforms = Compose([initialize_edge_weight])
    val_loaders, names = get_val_loaders(args.batch_size, my_transforms, num=num)
    print(f"\n===================\nRedo-views: {redo_views}\n===================\n")


    checkpoint = args.checkpoint

    if checkpoint == "latest":
        checkpoint_root = "wandb/latest-run/files"
        checkpoints = glob.glob(f"{checkpoint_root}/Checkpoint-*.pt")
        print(checkpoints)
        epochs = np.array([cp.split('-')[0] for cp in checkpoints])
        checkpoint_ind = np.argsort(epochs[::-1])[0]

        checkpoint_path = f"wandb/latest-run/files/{checkpoints[checkpoint_ind]}"

    elif checkpoint == "untrained":
        checkpoint_path = "untrained"

    else:
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

    # Retrieved saved models and load weights



    # model = GInfoMinMax(NodeEncoder(emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
    #                     proj_hidden_dim=args.emb_dim).to(device)
    #
    # if checkpoint != "untrained":
    #     model_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    #     model.load_state_dict(model_dict['encoder_state_dict'])
    #
    # general_ee = NodeEmbeddingEvaluation()
    # model.eval()
    # all_embeddings, separate_embeddings = general_ee.get_embeddings(model.encoder, val_loaders)

    metrics, metric_arrays, metric_names = get_metric_values(val_loaders)

    print(metrics, metric_arrays, metric_names)
    quit()
    # Get embeddings
    # general_ee = GeneralEmbeddingEvaluation()
    # model.eval()
    # all_embeddings, separate_embeddings = general_ee.get_embeddings(model.encoder, val_loaders)
    # embeddings_vs_metrics(val_loaders, all_embeddings, n_components=5, model_name=checkpoint[:-3])

    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(11,9))
    axes = [ax1, ax2, ax3, ax4]
    checkpoints = ["untrained", "chem-100.pt", "social-100.pt",  "all-100.pt"]
    # metrics, metric_arrays, metric_names = get_metric_values(val_loaders)
    # percentiles = [10., 10., 10., 0.5]
    for i_ax, ax in enumerate(axes):

        checkpoint = checkpoints[i_ax]
        model_name = checkpoint.split("-")[0]

        if checkpoint == "latest":
            checkpoint_root = "wandb/latest-run/files"
            checkpoints = glob.glob(f"{checkpoint_root}/Checkpoint-*.pt")
            print(checkpoints)
            epochs = np.array([cp.split('-')[0] for cp in checkpoints])
            checkpoint_ind = np.argsort(epochs[::-1])[0]

            checkpoint_path = f"wandb/latest-run/files/{checkpoints[checkpoint_ind]}"

        elif checkpoint == "untrained":
            checkpoint_path = "untrained"

        else:
            checkpoint_path = f"outputs/{checkpoint}"
            cfg_name = checkpoint.split('.')[0] + ".yaml"
            config_path = f"outputs/{cfg_name}"

            with open(config_path, 'r') as stream:
                try:
                    # Converts yaml document to python object
                    wandb_cfg = yaml.safe_load(stream)

                    # Printing dictionary
                    # print(wandb_cfg)
                except yaml.YAMLError as e:
                    # pass
                    print(e)

            args = wandb_cfg_to_actual_cfg(args, wandb_cfg)

        # Retrieved saved models and load weights

        model = GInfoMinMax(
            NodeEncoder(emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio,
                            pooling_type=args.pooling_type),
            proj_hidden_dim=args.emb_dim).to(device)

        view_learner = ViewLearner(
            NodeEncoder(emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio,
                            pooling_type=args.pooling_type),
            mlp_edge_model_dim=args.mlp_edge_model_dim).to(device)

        if checkpoint != "untrained":
            model_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(model_dict['encoder_state_dict'])
            view_learner.load_state_dict(model_dict['view_state_dict'])


        # Get embeddings
        general_ee = NodeEmbeddingEvaluation()
        model.eval()
        all_embeddings, separate_embeddings = general_ee.get_embeddings(model.encoder, val_loaders)
        #
        scaler = StandardScaler().fit(all_embeddings)
        print(f"Scaler means: {scaler.mean_.shape}, vars: {scaler.var_.shape}")
        all_embeddings = scaler.transform(all_embeddings)
        #
        print(f"All embeddings: {all_embeddings.shape}")

        # pca = PCA(n_components=10).fit(all_embeddings)
        # proj_all = pca.transform(all_embeddings)

        component_columns = [f"Component_{i}" for i in range(all_embeddings.shape[1])]

        dataset_names = []
        dataset_colors = []
        for iname, name in enumerate(names):
            dataset_names += [name] * separate_embeddings[iname].shape[0]
            dataset_colors += [iname / len(names)] * separate_embeddings[iname].shape[0]
        dataset_names = np.array(dataset_names)

        embedding_dict = {"Dataset": dataset_names[::10], "Dataset_cvalue": dataset_colors[:]}

        for icomp in range(6):
            embedding_dict[component_columns[icomp]] = all_embeddings[:, icomp]
        embedding_dataframe = pd.DataFrame(embedding_dict)
        embedding_dataframe.to_csv(f"outputs/embeddings_nodes_{model_name}.csv")

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

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()
    run(args)

