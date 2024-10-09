"""
Anonymous authors, 2023/24

Script for fine-tuning encoders on our validation datasets with full node and edge features.
This is achieved using FeaturedTransferModel, defined in unsupervised/encoder/featured_transfer_model.py.

Large sections are from the AD-GCL paper:

Susheel Suresh, Pan Li, Cong Hao, Georgia Tech, and Jennifer Neville. 2021.
Adversarial Graph Augmentation to Improve Graph Contrastive Learning.

In Advances in Neural Information Processing Systems,
Vol. 34. 15920–15933.
https://github.com/susheels/adgcl
"""


import argparse
import glob
import logging
import os
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import yaml
import wandb

from umap import UMAP
from sklearn.decomposition import PCA

from torch_geometric.transforms import Compose
from tqdm import tqdm

from utils import setup_wandb, wandb_cfg_to_actual_cfg, get_total_mol_onehot_dims
from datasets.loaders import get_val_loaders, get_test_loaders


from sklearn.metrics import roc_auc_score, mean_squared_error

from unsupervised.encoder import Encoder
from unsupervised.learning import GInfoMinMax
from unsupervised.utils import initialize_edge_weight
from unsupervised.encoder import FeaturedTransferModel
from torch.nn import MSELoss, BCELoss, Sigmoid

from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib


atom_feature_dims, bond_feature_dims = get_total_mol_onehot_dims()

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

def get_encoder(checkpoint, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint == "untrained":
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



    return checkpoint_path

def get_emb_y(loader, encoder, device, dtype='numpy', is_rand_label=False, every = 1, node_features = False):
	x, y = encoder.get_embeddings(loader, device, is_rand_label, every = every, node_features = node_features)
	if dtype == 'numpy':
		return x,y
	elif dtype == 'torch':
		return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
	else:
		raise NotImplementedError
    
def get_embeddings(encoder, loaders):
    encoder.eval()
    all_embeddings = None
    separate_embeddings = []
    # colours = []
    for i, loader in enumerate(tqdm(loaders, leave = False, desc = "Getting embeddings")):
        train_emb, train_y = get_emb_y(loader, encoder,
         "cuda" if torch.cuda.is_available() else "cpu",
          is_rand_label=False, every=1)

        separate_embeddings.append(train_emb)
        if all_embeddings is None:
            all_embeddings = train_emb
        else:
            all_embeddings = np.concatenate((all_embeddings, train_emb))
        # colours += [i for n in range(train_emb.shape[0])]

    return all_embeddings, separate_embeddings

def get_hdbscan_labels(embeddings):
    labels = HDBSCAN(n_jobs = 8).fit_predict(embeddings)

    return labels

def plot_projections(embedding_store, names, actual_names, mol_color_dict, color_dict):
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(8, 10))
    axes = [ax1, ax2, ax3, ax4]

    # Plot data
    handles_labels = []
    for i_check, separate_embeddings in enumerate(embedding_store):
        ax = axes[i_check]
        ax.set_title(actual_names[i_check])
        actual_name = actual_names[i_check]
        for i_emb, proj in enumerate(tqdm(separate_embeddings)):
            name = names[i_emb]
            if "ogb" in name:
                plot_marker = "^"
                color = mol_color_dict[name]
            else:
                plot_marker = "o"
                color = color_dict[name]
            scatter = ax.scatter(proj[:, 0], proj[:, 1], c=color, label=name, marker=plot_marker, s=2)
            handles_labels.append((scatter, name))
        # ax.set_title(f"{actual_name} (NC: {hdbscan_cluster_counts[i_check]})")
        ax.axis('off')


    # Get the legend handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Collect unique labels and handles
    handles, labels = zip(*handles_labels)

    unique_handles_labels = dict(zip(labels, handles)).items()
    unique_labels, unique_handles = zip(*unique_handles_labels)

    # Create a new legend with increased size for scatter points
    new_handles = []
    for ihandle, handle in enumerate(unique_handles):
        new_handle = plt.Line2D([0], [0],
            marker="^" if "ogb" in labels[ihandle] else "o", color='w',
             markerfacecolor=handle.get_facecolor()[0], markersize=30)
        new_handles.append(new_handle)

    # Add the legend
    fig.legend(new_handles, unique_labels,
                loc='lower left', bbox_to_anchor = (0.125, 0.),
                  ncol=4, frameon=False)
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
      # Adjust bottom to make space for the legend

    # Save the figure
    plt.savefig(f"outputs/everyone-pca.png", dpi=300)


def plot_projections_with_centroids(embedding_store, names, actual_names, mol_color_dict, color_dict):
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(8, 10))
    axes = [ax1, ax2, ax3, ax4]
    
    # Plot data
    handles_labels = []
    for i_check, separate_embeddings in enumerate(embedding_store):
        ax = axes[i_check]
        ax.set_title(actual_names[i_check])
        actual_name = actual_names[i_check]
        centroids = [np.mean(projection, axis = 0) for projection in separate_embeddings]
        for i_emb, proj in enumerate(tqdm(separate_embeddings)):
            name = names[i_emb]
            if "ogb" in name:
                plot_marker = "^"
                color = mol_color_dict[name]
            else:
                plot_marker = "o"
                color = color_dict[name]
            ax.scatter(proj[:, 0], proj[:, 1], c=color, label=name, marker=plot_marker, s=2, alpha = 0.05)
        # ax.set_title(f"{actual_name} (NC: {hdbscan_cluster_counts[i_check]})")
        ax.axis('off')

    for i_check, separate_embeddings in enumerate(embedding_store):
        ax = axes[i_check]
        centroids = [np.mean(projection, axis = 0) for projection in separate_embeddings]
        for i_emb, proj in enumerate(tqdm(separate_embeddings)):
            name = names[i_emb]
            if "ogb" in name:
                plot_marker = "^"
                color = mol_color_dict[name]
            else:
                plot_marker = "o"
                color = color_dict[name]
            scatter = ax.scatter(centroids[i_emb][0], centroids[i_emb][1],
                                  c=color, label=name, marker=plot_marker,
                                    s=100, linewidths = 2, edgecolors = "black", alpha = 1)
            handles_labels.append((scatter, name))
        # ax.set_title(f"{actual_name} (NC: {hdbscan_cluster_counts[i_check]})")
        ax.axis('off')

    



    # Get the legend handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Collect unique labels and handles
    handles, labels = zip(*handles_labels)

    unique_handles_labels = dict(zip(labels, handles)).items()
    unique_labels, unique_handles = zip(*unique_handles_labels)

    # Create a new legend with increased size for scatter points
    new_handles = []
    for ihandle, handle in enumerate(unique_handles):
        new_handle = plt.Line2D([0], [0],
            marker="^" if "ogb" in labels[ihandle] else "o", color='w',
             markerfacecolor=handle.get_facecolor()[0], markersize=30)
        new_handles.append(new_handle)

    # Add the legend
    fig.legend(new_handles, unique_labels,
                loc='lower left', bbox_to_anchor = (0.125, 0.),
                  ncol=4, frameon=False)
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
      # Adjust bottom to make space for the legend

    # Save the figure
    plt.savefig(f"outputs/everyone_centroids-pca.png", dpi=300)

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def centroid_similarities(embeddings, names, model_name = ""):
    embed_dim = embeddings[0].shape[1]
    centroids = np.zeros((len(embeddings), embed_dim))
    
    for i, embedding in enumerate(embeddings):
        assert np.sum(np.isnan(embedding)) == 0, f"Node features contain NaNs: {embedding}, dataset: {names[i]}, num nans {np.sum(np.isnan(embedding))}"
        centroids[i, :] = np.mean(embedding, axis = 0)

    pairwise_similarities = cosine_similarity(centroids)

    fig, ax = plt.subplots(figsize=(7,6))

    im = ax.imshow(pairwise_similarities, cmap = "binary", vmin = -1, vmax = 1)

    ax.set_xticks(np.arange(len(names)), labels = names)
    ax.set_yticks(np.arange(len(names)), labels = names)

    # for i1 in range(len(names)):
    # 	for i2 in range(len(names)):
    # 		text = ax.text(i2, i1, np.around(pairwise_similarities[i1, i2], decimals = 3),
    # 					   ha="center", va="center", color="w")

    annotate_heatmap(im, valfmt="{x:.3f}")

    plt.savefig(f"outputs/pairwise-similarity_{model_name}.png")

def run(args):
    """
    Run the transfer learning process for a given model checkpoint.

    Args:
        args: The command line arguments.

    Returns:
        None
    """


    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info(args)

    # setup_seed(args.seed)

    num = args.num
    evaluation_node_features = args.node_features
    print(f"\n\nNode features: {evaluation_node_features}\n\n")
    num_epochs = int(args.epochs)
    print(f"Num epochs: {num_epochs}")

    checkpoints = ["untrained","chem-100.pt",  "social-100.pt", "all-100.pt"]
    actual_names = ["GIN","ToP-Chem", "ToP-Social",  "ToP-All"]

    # Get datasets
    my_transforms = Compose([initialize_edge_weight])
    val_loaders, names = get_test_loaders(args.batch_size, my_transforms, num=num)
    # names = sorted(names)
    # val_loaders, names = get_val_loaders(args.batch_size, my_transforms, num=2*num)

    ogb_names, other_names = [], []
    for name in names:
        if "ogb" in name:
            ogb_names.append(name)
        else:
            other_names.append(name)

    cmap = plt.get_cmap('hsv')
    unique_categories = sorted(np.unique(other_names))
    colors = cmap(np.linspace(0.2, 0.8, int(len(unique_categories))))
    color_dict = {category: color for category, color in zip(unique_categories, colors)}

    cmap = plt.get_cmap('autumn')
    unique_categories = sorted(np.unique(ogb_names))
    colors = cmap(np.linspace(0, 1, len(unique_categories)))
    mol_color_dict = {category: color for category, color in zip(unique_categories, colors)}

    embedding_store = []
    hdbscan_cluster_counts = []

    for i_check, checkpoint in enumerate(checkpoints):


        device = "cuda" if torch.cuda.is_available() else "cpu"
        if checkpoint == "untrained":
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

        # checkpoint_path = get_encoder(checkpoint, args)
        model_name = checkpoint_path.split("/")[-1].split(".")[0]
        encoder = GInfoMinMax(Encoder(emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio,
        pooling_type=args.pooling_type, convolution=args.backbone), proj_hidden_dim=args.proj_dim).to(device)


        if checkpoint_path != "untrained":
            model_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            encoder.load_state_dict(model_dict['encoder_state_dict'], strict = False)
            print(f"Loaded state dict for {checkpoint_path}")
        actual_name = actual_names[i_check]

        all_embeddings, separate_embeddings = get_embeddings(encoder.encoder, val_loaders)

        centroid_similarities(separate_embeddings, names, model_name=actual_name)

        hdbscan_labels = get_hdbscan_labels(all_embeddings)
        
        n_clusters = np.unique(hdbscan_labels).shape[0]

        hdbscan_cluster_counts.append(n_clusters)

        fig, ax = plt.subplots(figsize = (6,7))
        ax.set_title(f"{actual_names[i_check]} (NC: {n_clusters})")

        ump = PCA(n_components=2).fit(all_embeddings) # UMAP(n_components = 2, n_neighbors=50, n_jobs=8).fit(all_embeddings)
        projections = []
        for i_emb, embedding in enumerate(tqdm(separate_embeddings)):
            proj = ump.transform(embedding)
            projections.append(proj)
            name = names[i_emb]

            if "ogb" in name:
                plot_marker = "^"
                color = mol_color_dict[name]
            else:
                plot_marker = "x"
                color = color_dict[name]

            ax.scatter(proj[:, 0], proj[:, 1], c = color, label = name, marker = plot_marker, s = 2)
        ax.axis('off')
        
        # Get the legend handles and labels
        handles, labels = ax.get_legend_handles_labels()

        # Create a new legend with increased size for scatter points
        new_handles = []
        for ihandle, handle in enumerate(handles):
            new_handle = plt.Line2D([0], [0],
             marker="^" if "ogb" in labels[ihandle] else "o", color='w', markerfacecolor=handle.get_facecolor()[0], markersize=30)
            new_handles.append(new_handle)

        # Add legend to the axis
        # ax.legend(handles=new_handles, labels=labels)

        ax.legend(handles = new_handles, labels = labels,
            shadow = True, bbox_to_anchor=(0.05, -0.2), loc='lower left', ncols = 3, frameon = False)

        plt.tight_layout()
        plt.savefig(f"outputs/{actual_name}-pca.png", dpi=300)

        embedding_store.append(projections)
    plot_projections_with_centroids(embedding_store, names, actual_names, mol_color_dict, color_dict)
    plot_projections(embedding_store, names, actual_names, mol_color_dict, color_dict)
    

    







def arg_parse():
    parser = argparse.ArgumentParser(description='AD-GCL ogbg-mol*')

    parser.add_argument('--dataset', type=str, default='ogbg-molesol',
                        help='Dataset')
    parser.add_argument('--model_lr', type=float, default=0.001,
                        help='Model Learning rate.')
    parser.add_argument('--view_lr', type=float, default=0.001,
                        help='View Learning rate.')
    parser.add_argument('--num_gc_layers', type=int, default=6,
                        help='Number of GNN layers before pooling')
    parser.add_argument('--pooling_type', type=str, default='standard',
                        help='GNN Pooling Type Standard/Layerwise')

    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimension')
    parser.add_argument('--proj_dim', type=int, default=300,
                        help='projection head dimension')

    parser.add_argument('--mlp_edge_model_dim', type=int, default=64,
                        help='embedding dimension')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.2,
                        help='Dropout Ratio / Probability')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Train Epochs')
    parser.add_argument('--reg_lambda', type=float, default=2.0, help='View Learner Edge Perturb Regularization Strength')

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--num', type=int, default=1000,
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

    parser.add_argument(
        '--backbone', type = str, default='gin', help = 'Model backbone to use (gin, gcn, gat)'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    run(args)