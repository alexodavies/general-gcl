"""
Very much work-in-progress
"""


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
import wandb

from torch_geometric.transforms import Compose, NormalizeFeatures, RandomLinkSplit
from tqdm import tqdm

from utils import better_to_nx, setup_wandb, wandb_cfg_to_actual_cfg, get_total_mol_onehot_dims
from datasets.loaders import get_train_loader, get_val_loaders, get_test_loaders


from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error, accuracy_score, precision_score, recall_score

from unsupervised.embedding_evaluation import GeneralEmbeddingEvaluation, TargetEvaluation
from unsupervised.encoder import Encoder
from unsupervised.learning import GInfoMinMax
from unsupervised.utils import initialize_edge_weight
from unsupervised.view_learner import ViewLearner
from unsupervised.encoder import TransferModel, FeaturedTransferModel, NodeClassificationTransferModel, EdgePredictionTransferModel

from torch.nn import MSELoss, BCELoss, Softmax, Sigmoid

from torch_geometric.datasets import Planetoid, Amazon, FacebookPagePage
from torch_geometric.data import DataLoader
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.sampler import NeighborSampler

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




def fine_tune(model, checkpoint_path, val_data, test_data, name, n_epochs):
    """
    Fine-tunes a given model on edge prediction using the provided data and hyperparameters.

    Args:
        model (torch.nn.Module): The model to be fine-tuned.
        checkpoint_path (str): The path to the checkpoint file for the model.
        val_data (LinkNeighborLoader): The validation data used for training.
        test_data (LinkNeighborLoader): The test data used for evaluation.
        name (str): The name of the model.
        n_epochs (int): The number of epochs to train the model.

    Returns:
        tuple: A tuple containing the following elements:
            - train_losses (list): A list of training losses for each epoch.
            - val_losses (list): A list of validation losses for each epoch.
            - max_train_score (float): The maximum training score achieved.
            - max_test_score (float): The maximum test score achieved.
            - max_val_loss (float): The maximum validation loss achieved.
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = checkpoint_path.split("/")[-1].split(".")[0]
    if model_name not in os.listdir("outputs"):
        os.mkdir(f"outputs/{model_name}")

    if checkpoint_path != "untrained":
        model_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_dict['encoder_state_dict'], strict=False)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    model.train()
    optimizer.zero_grad()

    score_fn = accuracy_score

    train_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[-1, -1],
        neg_sampling_ratio=2.0,
        batch_size=512,
        shuffle=True,
    )

    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=[-1,-1],
        neg_sampling_ratio=2.0,
        batch_size=512,
        shuffle=True,
    )


    train_losses, val_losses, scores = [], [], []
    for i_epoch in tqdm(range(n_epochs), leave=False, desc="Epoch"):
        model.train()
        epoch_train_losses, epoch_val_losses = [], []
        for ibatch, batch in enumerate(tqdm(train_loader, leave = False)):
            if ibatch > 100:
                continue
            optimizer.zero_grad()
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out.flatten(), batch.edge_label.to(device))

            loss.backward()  # Derive gradients.
            optimizer.step()

            epoch_train_losses.append(loss.item())


        train_losses.append(np.mean(epoch_train_losses))


    model.eval()
    with torch.no_grad():
        for ibatch, batch in enumerate(tqdm(test_loader, leave=False)):
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out.flatten(), batch.edge_label.to(device))

            score = score_fn(np.around(out.flatten().cpu().numpy(), decimals = 0), batch.edge_label.cpu().numpy())
            epoch_val_losses.append(loss.item())
    scores.append(score)
    val_losses.append(np.mean(epoch_val_losses))


    return train_losses, val_losses, max(scores), max(scores), max(val_losses)



def run(args):
    """
    Runs the edge prediction transfer process.

    Args:
        args: An object containing the command line arguments.

    Returns:
        None
    """
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info(args)

    num = args.num
    checkpoint = args.checkpoint
    evaluation_node_features = args.node_features
    print(f"\n\nNode features: {evaluation_node_features}\n\n")
    num_epochs = int(args.epochs)
    print(f"Num epochs: {num_epochs}")

    checkpoint_path = f"outputs/{checkpoint}"

    split_transform = RandomLinkSplit(
        num_val=0.05,
        num_test=0.05,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=False,
        is_undirected=True
    )


    # Get datasets
    my_transforms = Compose([initialize_edge_weight, NormalizeFeatures])
    datasets = []
    datasets += [Amazon(root='original_datasets/Amazon', name=name, transform=my_transforms) for name in ["Computers", "Photo"]]
    datasets += [Planetoid(root='original_datasets/Planetoid', name=name, transform=my_transforms) for name in ["Citeseer", "PubMed"]]

    splits = [split_transform(split.data) for split in datasets]

    train_splits = [split[0] for split in splits]
    val_splits = [split[1] for split in splits]
    test_splits = [split[2] for split in splits]
    names = ["Amazon-Computers", "Amazon-Photo",  "Citeseer", "PubMed"]


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

    model_name = checkpoint_path.split("/")[-1].split(".")[0]
    setup_wandb(args, name = "full-features-" + model_name + "-edge-prediction" if evaluation_node_features else model_name + "-edge-prediction", offline=True)
    wandb.log({"Transfer":True, "Edge_Transfer":True})
    wandb.log({"Model Name": model_name + "-features" if evaluation_node_features else model_name})

    for i in range(len(val_splits)):
        val_loader = train_splits[i]
        test_loader = test_splits[i]
        name = names[i]
        print(f"Name: {name}")

        if name not in os.listdir("outputs"):
            os.mkdir(f"outputs/{name}")
        n_repeats = 10

        best_pretrain_score = 0.
        best_untrain_score = 0.

        pretrain_val = np.zeros((n_repeats, num_epochs))
        pretrain_scores = []

        pbar = tqdm(range(n_repeats))
        for n in pbar:
            model = EdgePredictionTransferModel(
                Encoder(emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type, convolution=args.backbone),
                proj_hidden_dim=args.emb_dim, output_dim=1, features=evaluation_node_features,
                node_feature_dim=datasets[i].data.num_features if evaluation_node_features else 1, edge_feature_dim=1).to(device)

            pretrain_train_losses, pretrain_val_losses, pretrain_val_score, pretrain_best_epoch, pretrain_best_val_loss = fine_tune(model,
                                                                                                                                    checkpoint_path,
                                                                                                                                    val_loader,
                                                                                                                                    test_loader,
                                                                                                                                    name = name,
                                                                                                                                    n_epochs=num_epochs)

            pretrain_val[n, :] = pretrain_val_losses

            scores = "Pretrain:" + str(pretrain_val_score)[:5] # + "  Untrain:" + str(untrain_val_score)[:5] # + "default:  " +  str(default_val_score)[:5]
            pbar.set_description(scores)

            pretrain_scores.append(pretrain_val_score)

            if pretrain_val_score >= best_pretrain_score:
                best_pretrain_score = pretrain_val_score

        untrain_val_score = str(best_untrain_score)[:6]
        pretrain_val_score = str(best_pretrain_score)[:6]

        pretrain_val_loss_mean = np.mean(pretrain_val, axis=0)

        pretrain_val_loss_max = np.max(pretrain_val, axis=0)

        pretrain_val_loss_min = np.min(pretrain_val, axis=0)

        pretrain_mean_score = str(np.mean(pretrain_scores))[:5]

        pretrain_dev_score = str(np.std(pretrain_scores))[:5]


        wandb.log({f"{name}-edge-pred/model-mean": float(pretrain_mean_score),
                   f"{name}-edge-pred/model-dev": float(pretrain_dev_score),
                   f"{name}-edge-pred/model-best": float(pretrain_val_score)})


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
    parser.add_argument('--epochs', type=int, default=10,
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

