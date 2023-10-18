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

import wandb
from utils import better_to_nx, vis_from_pyg, vis_grid, setup_wandb
from datasets.loaders import get_train_loader, get_val_loaders, get_test_loaders

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
               average_degree, nx.average_clustering, nx.transitivity]

    # Compute metrics for all graphs
    metric_arrays = [np.array([metric(g) for g in tqdm(val_data)]) for metric in metrics]

    # 5-cycles is multi-processed, so needs to be handled differently
    # metric_arrays += [np.array(five_cycles(val_data))]
    # metrics += ["five-cycles"]

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





def vis_vals(loader, dataset_name, num = 10000):
    """
    Visualise samples from a dataloader
    Args:
        loader: dataloader for the respective dataset
        dataset_name: name under which to save images - should be in original_datasets
        num: number of images to produce

    Returns:
        None
    """
    data_directory = os.getcwd() + '/original_datasets/' + dataset_name
    if "vals" not in os.listdir(data_directory):
        print(f"Making {os.getcwd() + '/original_datasets/' + dataset_name + '/vals'}")
        os.mkdir(os.getcwd() + '/original_datasets/' + dataset_name + '/vals')

    existing_files = os.listdir(os.getcwd() + '/original_datasets/' + dataset_name + f'/vals')
    skip_all = True
    for i in range(num):

        if not skip_all:
            break

        if f"val-{i}.png" in existing_files:
            pass
        else:
            skip_all = False
            break

    if skip_all:
        return

    val_data = []
    for batch in loader:
        val_data += batch.to_data_list()

    existing_files = os.listdir(os.getcwd() + '/original_datasets/' + dataset_name + f'/vals')

    vis_grid(val_data[:16], os.getcwd() + '/original_datasets/' + dataset_name + f'/val-grid-{dataset_name}.png')

    for i, data in enumerate(tqdm(val_data)):
        filename = os.getcwd() + '/original_datasets/' + dataset_name + f'/vals/val-{i}.png'
        if i >= num or f"val-{i}.png" in existing_files:
            pass
        else:
            vis_from_pyg(data, filename=filename)

def vis_views(view_learner, loader, dataset_name, num = 10000, force_redo = False):
    """
    Visualise views/augmentations of samples in a dataloader
    Args:
        view_learner: the view learner (duh)
        loader: dataloader for the respective dataset
        dataset_name: name under which to save images - should be in original_datasets
        num: number of images to produce
        force_redo: whether to re-produce images that already exist (ie for a different view learner)

    Returns:
        None
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_directory = os.getcwd() + '/original_datasets/' + dataset_name

    if "views" not in os.listdir(data_directory):
        print(f"Making {os.getcwd() + '/original_datasets/' + dataset_name + '/views'}")
        os.mkdir(os.getcwd() + '/original_datasets/' + dataset_name + '/views')


    existing_files = os.listdir(os.getcwd() + '/original_datasets/' + dataset_name + f'/views')
    skip_all = True if not force_redo else False
    for i in range(min(len(loader), num)):

        if not skip_all:
            break

        if f"view-{i}.png" in existing_files or not skip_all:
            pass
        else:
            skip_all = False
            break

    if skip_all:
        return

    view_data = []
    total_edges, total_dropped = 0, 0

    view_learner.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, batch.edge_attr)

            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(device)
            gate_inputs = (gate_inputs + edge_logits) / temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

            kept = torch.bernoulli(batch_aug_edge_weight).to(torch.bool)

            total_edges += kept.shape[0]
            total_dropped += kept.shape[0] - torch.sum(kept)
            # Kept is a boolean array of whether an edge is kept after the view
            datalist = batch.to_data_list()
            edges_so_far = 0
            for idata, data in enumerate(datalist):

                dropped_slice = kept[edges_so_far:edges_so_far + data.num_edges]
                new_edges = data.edge_index[:,dropped_slice]
                edges_so_far += data.num_edges
                data.edge_index = new_edges
                datalist[idata] = data

            view_data += datalist


    percent_dropped = str(100*int(total_dropped)/int(total_edges))[:5]
    print(f"Dropped {percent_dropped}% of edges")
    vis_grid(view_data[:16], os.getcwd() + '/original_datasets/' + dataset_name + f'/view-grid-{dataset_name}.png')

    existing_files = os.listdir(os.getcwd() + '/original_datasets/' + dataset_name + f'/views')
    for i, data in enumerate(tqdm(view_data)):
        filename = os.getcwd() + '/original_datasets/' + dataset_name + f'/views/view-{i}.png'


        if force_redo and i <= num:
            vis_from_pyg(data, filename=filename)

        elif i >= num or f"view-{i}.png" in existing_files:
            # print(f"{filename} already exists")
            pass

        else:
            vis_from_pyg(data, filename=filename)

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
        pass

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
    val_loaders, names = get_val_loaders(args.batch_size, my_transforms, num=num)

    # Visualise
    for i, loader in enumerate(val_loaders):
        vis_vals(loader, names[i], num = num)
        vis_views(view_learner, loader, names[i], num=num, force_redo=redo_views)

    # Get embeddings
    general_ee = GeneralEmbeddingEvaluation()
    model.eval()
    all_embeddings, separate_embeddings = general_ee.get_embeddings(model.encoder, val_loaders)
    embeddings_vs_metrics(val_loaders, all_embeddings, n_components=10)


    # embedder = ComponentSlicer(comp_1=10, comp_2=20)
    # embedder = Isomap(n_components=2, n_neighbors=250, n_jobs=6).fit(all_embeddings)
    embedder = TruncatedSVD(n_components=10).fit(all_embeddings)
    # embedder = UMAP(n_components=2, n_neighbors=50, n_jobs=6)
    # embedder.fit(all_embeddings)

    #Prepare data for bokeh dashboard
    x, y, plot_names, plot_paths, view_paths  = [], [], [], [], []
    ind_x, ind_y, ind_plot_names, ind_plot_paths, ind_view_paths = [], [], [], [], []

    for i, emb in enumerate(separate_embeddings):
        proj = embedder.transform(emb)

        proj_x, proj_y = proj[:num,0].tolist(), proj[:num,1].tolist()
        x += proj_x
        y += proj_y

        ind_x.append(proj_x)
        ind_y.append(proj_y)

        plot_names += len(proj_x)*[names[i]]

        ind_plot_names.append(names[i])

        img_root = os.getcwd() + '/original_datasets/' + names[i] + '/vals/*.png'
        img_paths = glob.glob(img_root)

        filenames = [int(filename.split('/')[-1][4:-4]) for filename in img_paths]
        sort_inds = np.argsort(filenames).tolist()
        plot_paths += [img_paths[ind] for ind in sort_inds][:num]

        ind_plot_paths.append([img_paths[ind] for ind in sort_inds][:num])


        img_root = os.getcwd() + '/original_datasets/' + names[i] + '/views/*.png'
        img_paths = glob.glob(img_root)

        filenames = [int(filename.split('/')[-1][5:-4]) for filename in img_paths]
        sort_inds = np.argsort(filenames).tolist()
        view_paths += [img_paths[ind] for ind in sort_inds]

        ind_view_paths.append([img_paths[ind] for ind in sort_inds])

    hover = HoverTool(
        tooltips="""
            <div>
                <div>
                    <span style="font-size: 15px; font-weight: bold;">@desc</span>
                    <span style="font-size: 10px;">Original and View</span>
                </div>
                <div>
                    <img
                        src="@imgs" height="128" alt="missing" width="128"
                        style="float: left; margin: 0px 0px 0px 0px;"
                        border="2"
                    ></img>
                    <img
                        src="@views" height="128" alt="missing" width="128"
                        style="float: left; margin: 0px 0px 0px 0px;"
                        border="2"
                    ></img>
                </div>

            </div>
            """
    )

    p = figure(tools=[hover, PanTool(), BoxZoomTool(), WheelZoomTool(), UndoTool(), ResetTool()],
                aspect_ratio = 16/8, lod_factor = 10)

    unique_names = np.arange(len(names))

    colors = [
        "#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255 * mpl.cm.tab20(mpl.colors.Normalize()(unique_names))
    ]

    for i in range(len(names)):
        print(names[i], len(ind_x[i]), len(ind_plot_paths[i]), len(ind_view_paths[i]))
        name = names[i]
        source = ColumnDataSource(
            data=dict(
                x=ind_x[i],
                y=ind_y[i],
                desc= len(ind_x[i]) * [name],
                imgs=ind_plot_paths[i],
                views=ind_view_paths[i]
            )
        )

        p.scatter("x", "y", color = colors[i], size=5, alpha=1, legend_label=name, source=source)
    p.legend.click_policy = "hide"
    p.legend.location = "top_left"
    p.sizing_mode = 'scale_width'


    x_array, y_array = np.array(sorted(x)), np.array(sorted(y))
    n_points = x_array.shape[0]
    outlier_x = [x_array[int(0.01*n_points)],x_array[int(0.99*n_points)]]
    outlier_y = [y_array[int(0.01*n_points)], y_array[int(0.99*n_points)]]

    upper_lim_x, lower_lim_x = max(outlier_x), min(outlier_x)
    upper_lim_y, lower_lim_y = max(outlier_y), min(outlier_y)

    p.x_range = Range1d(lower_lim_x, upper_lim_x)
    p.y_range = Range1d(lower_lim_y, upper_lim_y)

    p.xaxis.axis_label = f"{embedder} 0" # {embedder.explained_variance_ratio_[0]}"
    p.yaxis.axis_label = f"{embedder} 1" # {embedder.explained_variance_ratio_[1]}"

    output_file("bokeh-embedding-dashboard.html")
    show(p)




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

