"""
Anonymous authors, 2023/24

Script for fine-tuning encoders on node-classification tasks

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
import numpy as np
import torch
import yaml
import wandb

from torch_geometric.transforms import Compose, NormalizeFeatures
from tqdm import tqdm

from utils import setup_wandb, wandb_cfg_to_actual_cfg, get_total_mol_onehot_dims

from sklearn.metrics import accuracy_score

from unsupervised.encoder import Encoder
from unsupervised.utils import initialize_edge_weight
from unsupervised.encoder import NodeClassificationTransferModel

from torch_geometric.datasets import Planetoid, GitHub, LastFMAsia
from sklearn.model_selection import train_test_split
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import Data

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

def generate_node_masks(data, test_size=0.2, random_state=None):
    """
    Generate train and test masks for a Torch Geometric Data object.

    Args:
        data (Data): Torch Geometric Data object.
        test_size (float, optional): Proportion of the dataset to include in the test split.
                                      Defaults to 0.2.
        random_state (int, optional): Seed used by the random number generator.
                                      Defaults to None.

    Returns:
        tuple: Tuple containing train and test masks.
    """
    # Splitting your data into train and test sets
    num_nodes = data.x.shape[0]
    train_idx, test_idx = train_test_split(range(num_nodes), test_size=test_size, random_state=random_state)

    # Generating node masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True

    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_idx] = True

    return train_mask, test_mask


def create_data_loaders(graph, train_ratio=0.8, batch_size=64, num_neighbors=[10]):
    """
    Create train and test data loaders for node classification tasks.

    Parameters:
        graph (torch_geometric.data.Data): The input graph data object.
        train_ratio (float, optional): The ratio of nodes to be included in the training set.
                                       Defaults to 0.8.
        batch_size (int, optional): The batch size for data loading. Defaults to 64.
        num_neighbors (list of int, optional): The number of neighbors to sample for each node.
                                                Defaults to [10].

    Returns:
        train_loader (LinkNeighborLoader): Data loader for the training set.
        test_loader (LinkNeighborLoader): Data loader for the test set.

    Notes:
        - The input graph should be an instance of torch_geometric.data.Data.
        - The function creates train and test masks based on the specified train_ratio.
        - It initializes LinkNeighborLoader objects for both training and testing data,
          which are used for neighbor sampling during node classification tasks.
        - LinkNeighborLoader is a custom data loader designed for node classification tasks,
          capable of sampling node features and their corresponding neighbors efficiently.

    Example:
        # Assuming 'graph' is a torch_geometric.data.Data object
        train_loader, test_loader = create_data_loaders(graph, train_ratio=0.8,
                                                         batch_size=64, num_neighbors=[10])
    """

    # Convert to undirected graph
    # graph = to_undirected(graph)

    # Split edge indices
    edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
    edge_attr = graph.edge_attr if hasattr(graph, 'edge_attr') else None

    # Create a Data object
    data = Data(edge_index=edge_index, edge_attr=edge_attr, x=graph.x, y=graph.y)

    # Create train and test masks
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[:int(train_ratio * data.num_nodes)] = 1

    # Create LinkNeighborLoader
    train_loader = LinkNeighborLoader(
        data, num_neighbors=num_neighbors, batch_size=batch_size, shuffle=True
    )
    test_loader = LinkNeighborLoader(
        data, num_neighbors=num_neighbors, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader

def fine_tune(model, checkpoint_path, dataset, name, n_epochs):
    """
    Fine-tune a given model using transfer learning on a dataset for node classification tasks.

    Parameters:
        model (torch.nn.Module): The model to be fine-tuned.
        checkpoint_path (str): The path to the checkpoint file for loading pre-trained weights.
                               If "untrained", the model is initialized without pre-trained weights.
        dataset (torch_geometric.datasets): The dataset for node classification.
        name (str): The name of the model for saving checkpoints and results.
        n_epochs (int): The number of epochs for fine-tuning.

    Returns:
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
        max_train_score (float): Maximum accuracy achieved on the training set during fine-tuning.
        max_val_score (float): Maximum accuracy achieved on the validation set during fine-tuning.
        min_val_loss (float): Minimum loss achieved on the validation set during fine-tuning.

    Notes:
        - The function fine-tunes a given model on a dataset for node classification tasks.
        - If a checkpoint path is provided, pre-trained weights are loaded into the model.
        - The training and validation datasets are split using a train-test split ratio defined in the dataset.
        - The model is trained using the Adam optimizer and cross-entropy loss.
        - Training progress is monitored using training and validation losses and accuracy scores.
        - The function returns lists of training losses, validation losses, and accuracy scores,
          along with the maximum training and validation accuracy scores and the minimum validation loss.
        - The model checkpoints and results are saved with the specified name for further analysis.

    Example:
        # Assuming 'model' is a torch.nn.Module, 'checkpoint_path' points to a pre-trained checkpoint,
        # 'dataset' is a torch_geometric.datasets object, 'name' is a string, and 'n_epochs' is an integer.
        train_losses, val_losses, max_train_score, max_val_score, min_val_loss = fine_tune(
            model, checkpoint_path, dataset, name, n_epochs
        )
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = checkpoint_path.split("/")[-1].split(".")[0]
    if model_name not in os.listdir("outputs"):
        os.mkdir(f"outputs/{model_name}")

    if checkpoint_path != "untrained":
        model_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_dict['encoder_state_dict'], strict=False)

    model.to(device)
    train_loader, test_loader = create_data_loaders(dataset.data, num_neighbors=[5,5,5])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    optimizer.zero_grad()

    train_losses, val_losses, scores = [], [], []
    for i_epoch in tqdm(range(n_epochs), leave=False):
        epoch_train_losses = []
        for ibatch, batch in enumerate(tqdm(train_loader, leave=False)):
            model.train()
            optimizer.zero_grad()

            # Use train mask to filter training nodes and their labels
            train_batch = batch.to(device)
            train_mask = train_batch.train_mask
            out = model(train_batch.x, train_batch.edge_index,
                        torch.ones(train_batch.edge_index.shape[1]).reshape(-1, 1).to(device))[0]
            loss = criterion(out[train_mask], train_batch.y[train_mask].to(device))
            epoch_train_losses.append(loss.item())
            loss.backward()  # Derive gradients.
            optimizer.step()


        train_losses.append(np.mean(epoch_train_losses))

    model.eval()
    epoch_scores, epoch_test_losses = [], []
    for ibatch, batch in enumerate(tqdm(test_loader, leave=False)):
        # No need to use train_mask during validation
        val_batch = batch.to(device)
        test_mask = ~val_batch.train_mask
        out = model(val_batch.x, val_batch.edge_index,
                    torch.ones(val_batch.edge_index.shape[1]).reshape(-1, 1).to(device))[0]
        val_loss = criterion(out[test_mask], val_batch.y.to(device)[test_mask])
        epoch_test_losses.append(val_loss.item())
        preds = np.argmax(out[test_mask].detach().cpu().numpy(), axis=1)
        epoch_scores.append(accuracy_score(preds, batch.y[test_mask].cpu().numpy()))

    val_losses.append(np.mean(epoch_test_losses))
    scores.append(np.mean(epoch_scores))

    return train_losses, val_losses, max(scores), max(scores), min(val_losses)





def run(args):
    """
    Run node classification transfer learning experiments using pre-trained models.

    Parameters:
        args (argparse.Namespace): Command-line arguments containing experiment configuration.

    Returns:
        None

    Notes:
        - This function orchestrates the process of running node classification transfer learning experiments.
        - It initializes the experiment environment, including setting up logging and determining the device for computation.
        - Depending on the provided arguments, it loads pre-trained models or initializes untrained models.
        - It sets up datasets for validation, including GitHub, LastFMAsia, and Planetoid datasets.
        - The function conducts experiments with multiple repetitions, fine-tuning models and evaluating their performance.
        - It logs the results using Weights & Biases (wandb) for monitoring and analysis.
    """
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info(args)

    checkpoint = args.checkpoint
    evaluation_node_features = args.node_features
    print(f"\n\nNode features: {evaluation_node_features}\n\n")
    num_epochs = int(args.epochs)
    print(f"Num epochs: {num_epochs}")

    # Get datasets
    my_transforms = Compose([initialize_edge_weight, NormalizeFeatures()])

    val_loaders = []
    val_loaders += [
                    GitHub(root='original_datasets/GitHub', transform=my_transforms),
                    LastFMAsia(root='original_datasets/LastFM', transform=my_transforms)]
    val_loaders += [Planetoid(root='original_datasets/Planetoid', name=name, transform=my_transforms) for name in ["Citeseer", "PubMed"]]
    names = ["GitHub", "LastFM",  "Citeseer", "PubMed"]

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
    setup_wandb(args,
                name = "full-features-" + model_name + "-node-classification" if evaluation_node_features else model_name + "-node-classification",
                offline=True)
    wandb.log({"Transfer":True, "Node_Transfer":True})
    wandb.log({"Model Name": model_name + "-features" if evaluation_node_features else model_name})


    for i in range(len(val_loaders)):
        val_loader = val_loaders[i]
        name = names[i]
        print(f"Name: {name}")

        if name not in os.listdir("outputs"):
            os.mkdir(f"outputs/{name}")

        n_repeats = 10

        best_pretrain_score = 0.

        pretrain_val = np.zeros((n_repeats, num_epochs))
        pretrain_scores = []

        pbar = tqdm(range(n_repeats))
        for n in pbar:
            model = NodeClassificationTransferModel(
                Encoder(emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type, convolution = args.backbone),
                proj_hidden_dim=args.emb_dim, output_dim=val_loader.num_classes, features=evaluation_node_features,
                node_feature_dim=val_loader.num_features if evaluation_node_features else 1, edge_feature_dim=1).to(device)

            pretrain_train_losses, pretrain_val_losses, pretrain_val_score, pretrain_best_epoch, pretrain_best_val_loss = fine_tune(model,
                                                                                                                                    checkpoint_path,
                                                                                                                                    val_loader,
                                                                                                                                    name = name,
                                                                                                                                    n_epochs=num_epochs)

            pretrain_val[n, :] = pretrain_val_losses

            scores = "Pretrain:" + str(pretrain_val_score)[:5] #  + "  Untrain:" + str(untrain_val_score)[:5] # + "default:  " +  str(default_val_score)[:5]
            pbar.set_description(scores)

            pretrain_scores.append(pretrain_val_score)

            if pretrain_val_score >= best_pretrain_score:
                best_pretrain_score = pretrain_val_score

        pretrain_val_score = str(best_pretrain_score)[:6]

        pretrain_mean_score = str(np.mean(pretrain_scores))[:5]

        pretrain_dev_score = str(np.std(pretrain_scores))[:5]

        wandb.log({f"{name}-node-class/model-mean": float(pretrain_mean_score),
                   f"{name}-node-class/model-dev": float(pretrain_dev_score),
                   f"{name}-node-class/model-best": float(pretrain_val_score)})

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
    parser.add_argument('--mlp_edge_model_dim', type=int, default=32,
                        help='embedding dimension')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.2,
                        help='Dropout Ratio / Probability')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Train Epochs')
    parser.add_argument('--reg_lambda', type=float, default=5.0, help='View Learner Edge Perturb Regularization Strength')

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

