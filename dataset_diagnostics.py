import argparse
import concurrent.futures
import logging
import random
import networkx as nx
import numpy as np
import torch
from torch_geometric.transforms import Compose
from tqdm import tqdm
from utils import better_to_nx
from datasets.loaders import get_train_datasets, get_val_datasets, get_test_datasets
from unsupervised.utils import initialize_edge_weight


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


def prettify_metric_name(metric):
    try:
        metric_name = str(metric).split(' ')[1]
    except:
        metric_name = str(metric)
    # metrics = [nx.number_of_nodes, nx.number_of_edges, nx.density, safe_diameter,
    #            average_degree, nx.average_clustering, nx.transitivity]
    pretty_dict = {"number_of_nodes": "Num. Nodes",
                   "number_of_edges": "Num. Edges",
                   "density": "Density",
                   "safe_diameter": "Diameter",
                   "average_degree": "Avg. Degree",
                   "average_clustering": "Avg. Clust.",
                   "transitivity": "Trans.",
                   "three_cycles": "Num. 3-Cycles",
                   "four_cycles":"Num. 4-Cycles"}

    return pretty_dict[metric_name]

def clean_graph(g):
    Gcc = sorted(nx.connected_components(g), key=len, reverse=True)
    g = g.subgraph(Gcc[0]).copy()
    g.remove_edges_from(nx.selfloop_edges(g))

    return g

def get_metric_values(dataset):


    val_data = [better_to_nx(data)[0] for data in dataset]
    for i, item in enumerate(val_data):
        # item.remove_edges_from(nx.selfloop_edges(item))
        val_data[i] = clean_graph(item)

    # Metrics can be added here - should take an nx graph as input and return a numerical value
    metrics = [nx.number_of_nodes, nx.number_of_edges, safe_diameter,
               nx.average_clustering,] #, average_degree, ]
    metric_names = [prettify_metric_name(metric) for metric in metrics]
    # Compute metrics for all graphs
    metric_arrays = [np.array([metric(g) for g in tqdm(val_data, leave=False, desc=metric_names[i_metric])]) for i_metric, metric in enumerate(metrics)]

    return metric_arrays, metrics,  metric_names






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

    # Get datasets
    my_transforms = Compose([initialize_edge_weight])

    train_datasets, train_names = get_train_datasets(my_transforms)
    val_datasets, val_names = get_val_datasets(my_transforms)
    test_datasets, _ = get_test_datasets(my_transforms)


    # for i_dataset, dataset in enumerate(train_datasets):
    #     arrays, metrics, names = get_metric_values(dataset)
    #     if i_dataset == 0:
    #         print_string = " & Split & Num. Graphs "
    #         for name in names:
    #             print_string += f"& {name}"
    #         print(print_string + r"\\")
    #
    #
    #     print_string = f"{train_names[i_dataset]} & Train & {len(dataset)} "
    #     for i_name, name in enumerate(names):
    #         value, dev = np.mean(arrays[i_name]), np.std(arrays[i_name])
    #         value = float('%.3g' % value)
    #         dev = float('%.3g' % dev)
    #         print_string += f"& {value} $\pm$ {dev} "
    #     print(print_string + r"\\")

    print("\n\n")

    for i_dataset, dataset in enumerate(val_datasets):
        arrays, metrics, names = get_metric_values(dataset)
        if "ogbg" not in val_names[i_dataset]:
            continue
        print_string = f"{val_names[i_dataset]} & Val & {len(dataset)}"
        for i_name, name in enumerate(names):
            value, dev = np.mean(arrays[i_name]), np.std(arrays[i_name])
            value = float('%.3g' % value)
            dev = float('%.3g' % dev)
            print_string += f"& {value} $\pm$ {dev}"
        print(print_string + r"\\")

    print("\n\n")

    for i_dataset, dataset in enumerate(test_datasets):
        arrays, metrics, names = get_metric_values(dataset)
        if "ogbg" not in val_names[i_dataset]:
            continue
        print_string = f"{val_names[i_dataset]} & Test & {len(dataset)}"
        for i_name, name in enumerate(names):
            value, dev = np.mean(arrays[i_name]), np.std(arrays[i_name])
            value = float('%.3g' % value)
            dev = float('%.3g' % dev)
            print_string += f"& {value} $\pm$ {dev}"
        print(print_string + r"\\")
    # train_arrays, metrics,  metric_names = get_metric_values(train_datasets)
    # val_arrays, _, __ = get_metric_values(val_datasets)
    # test_arrays, _, __ = get_metric_values(test_datasets)

    # by_dataset_metrics = {metric_names[i]:[train_arrays[i], val_arrays[i], test_arrays[i]] for i in range(len(train_arrays))}












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

