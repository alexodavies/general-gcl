import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch_geometric
from tqdm import tqdm
import copy
from datetime import datetime
from torch_geometric.data import DataLoader
from matplotlib.colors import LogNorm

from datasets.loaders import get_chemical_datasets, get_val_loaders, get_test_loaders

from noisenoise import add_weighted_noise_to_dataset, compute_onehot_probabilities, compute_onehot_probabilities_edge, add_noise_to_dataset
from unsupervised.utils import initialize_edge_weight
from torch_geometric.transforms import Compose


from unsupervised.encoder import Encoder
from unsupervised.encoder import FeaturedTransferModel
from torch.nn import MSELoss, BCELoss, Sigmoid

import wandb

from sklearn.metrics import roc_auc_score, mean_squared_error

import os

from utils import get_total_mol_onehot_dims
from features_transfer import arg_parse
atom_feature_dims, bond_feature_dims = get_total_mol_onehot_dims()

def setup_wandb(cfg, offline = False, name = None):
    """
    Uses a config dictionary to initialise wandb to track sampling.
    Requires a wandb account, https://wandb.ai/

    params: cfg: argparse Namespace

    returns:
    param: cfg: same config
    """
    print(os.getcwd())
    kwargs = {'name': name if name is not None else 'all' + datetime.now().strftime("%m-%d-%Y-%H-%M-%S"),
               'project': f'noise-noise-molecules',
                 'config': cfg,
              'reinit': True, 'entity':'hierarchical-diffusion',
              'mode':'online' if offline else 'online'}
    wandb.init(**kwargs)
    wandb.save('*.txt')

    return cfg


def tidy_labels(labels):
    """
    Tidies up the given labels by converting them into a consistent format.

    Args:
        labels (list or numpy.ndarray): The input labels to be tidied.

    Returns:
        numpy.ndarray: The tidied labels.

    Raises:
        None

    """

    if type(labels[0]) is not list:
        if np.sum(labels) == np.sum(np.array(labels).astype(int)):
            labels = np.array(labels).astype(int)
        else:
            labels = np.array(labels)
        return labels

    # Could be lists of floats
    elif type(labels[0][0]) is float:
        return np.array(labels)

    # Possibility of one-hot labels
    elif np.sum(labels[0][0]) == 1 and type(labels[0][0]) is int:

        new_labels = []
        for label in labels:
            new_labels.append(np.argmax(label))

        return np.array(new_labels)

    else:
        return np.array(labels)

def get_task_type(loader, name):
    """
    Determine the task type based on the data loader and dataset name.

    Args:
        loader (torch.utils.data.DataLoader): The data loader.
        name (str): The name of the dataset.

    Returns:
        str: The task type, which can be "empty", "classification", or "regression".
    """

    val_targets = []
    for i_batch, batch in enumerate(loader):
        if batch.y is None or name == "ogbg-molpcba" or name == "blank":
            task = "empty"
            n_samples = 0
            return task

        else:
            selected_y = batch.y
            if type(selected_y) is list:
                selected_y = torch.Tensor(selected_y)

            if selected_y.dim() > 1:
                selected_y = [selected_y[i, :].cpu().numpy().tolist() for i in range(selected_y.shape[0])]
            else:
                selected_y = selected_y.cpu().numpy().tolist()

            val_targets += selected_y

        break

    val_targets = tidy_labels(val_targets).flatten()
    if type(val_targets[0]) is int or type(val_targets[0]) is np.int64:
        task = "classification"
    else:
        task = "regression"

    return task

def evaluate_model(model, test_loader, score_fn, out_fn, loss_fn, task):
    """
    Evaluates a model on a test dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (torch.utils.data.DataLoader): The test data loader.
        score_fn (callable): The scoring function to evaluate the model's predictions.
        out_fn (callable): The output function to apply to the model's predictions.
        loss_fn (callable): The loss function to calculate the model's loss.
        task (str): The task type, either "classification" or "regression".

    Returns:
        tuple: A tuple containing the evaluation score and the loss.

    Raises:
        Exception: If an error occurs during evaluation.

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    with torch.no_grad():
        ys, y_preds = [], []
        for batch in test_loader:
            batch.to(device)
            y_pred, _ = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, None)
            y = batch.y

            if type(y) is list:
                y = torch.tensor(y).to(device)

            if task == "classification":
                y_pred = out_fn(y_pred).flatten()
                if y.dim() > 1:
                    y = y[:, 0]
                y = y.to(y_pred.dtype)

            y_pred = y_pred.cpu().numpy().tolist()
            y = y.cpu().numpy().tolist()

            ys += y
            y_preds += y_pred
    model.train()
    try:
        return score_fn(ys, y_preds, squared=False), loss_fn(torch.tensor(ys), torch.tensor(y_preds))
    except:
        return score_fn(ys, y_preds), loss_fn(torch.tensor(ys), torch.tensor(y_preds))


def fine_tune(model, checkpoint_path, val_loader, test_loader, name="blank", n_epochs=50):
    """
    Fine-tunes a given model using the provided data loaders and hyperparameters.

    Args:
        model (torch.nn.Module): The model to be fine-tuned.
        checkpoint_path (str): The path to the checkpoint file for the model.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation set.
        test_loader (torch.utils.data.DataLoader): The data loader for the test set.
        name (str, optional): The name of the model. Defaults to "blank".
        n_epochs (int, optional): The number of epochs for training. Defaults to 50.

    Returns:
        tuple: A tuple containing the following elements:
            - train_losses (list): The list of training losses for each epoch.
            - val_losses (list): The list of validation losses for each epoch.
            - final_score (float): The final score of the model on the test set.
            - best_epoch (int): The epoch number with the best validation loss.
            - best_val_loss (float): The best validation loss achieved during training.
    """

    # At the moment this is rigid to single-value predictions

    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = get_task_type(val_loader, name)

    model_name = checkpoint_path.split("/")[-1].split(".")[0]
    if model_name not in os.listdir("outputs"):
        os.mkdir(f"outputs/{model_name}")

    if task == "empty":
        return

    if task == "classification":
        loss_fn = BCELoss()
        score_fn = roc_auc_score
    else:
        loss_fn = MSELoss()
        score_fn = mean_squared_error


    if "untrained" not in checkpoint_path:
        model_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_dict['encoder_state_dict'], strict=False)
    model.to(device)

    out_fn = Sigmoid()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)
    # pbar = tqdm(range(n_epochs), leave=False)
    best_val_loss, best_epoch = 1.e9, 0
    train_losses, val_losses = [], []
    for i_epoch in range(n_epochs):
        model.train()
        # ys, y_preds = [], []
        for i_batch, batch in enumerate(val_loader):
            model.zero_grad()
            # set up
            batch = batch.to(device)
            y = batch.y

            if type(y) is list:
                y = torch.tensor(y).to(device)

            y_pred, _ = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, None)
            if task == "classification":
                y_pred = out_fn(y_pred).flatten()
                if y.dim() > 1:
                    y = y[:, 0]
            y = y.to(y_pred.dtype)

            model_loss = loss_fn(y_pred, y)
            train_losses.append(model_loss.item())

            model_loss.backward()
            model_optimizer.step()
        # pbar.set_description(str(model_loss.item())[:6])
        val_score, val_loss = evaluate_model(model, test_loader, score_fn, out_fn, loss_fn, task)

        wandb.log({f"{name}/Val-Loss":val_loss.item(),
                   f"{name}/Val-Score":val_score})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = i_epoch
            final_score = evaluate_model(model, test_loader, score_fn, out_fn, loss_fn, task)[0]

        val_losses.append(val_loss.item())

    if task == "classification":
        final_score = 1 - final_score

    return train_losses, val_losses, final_score, best_epoch, best_val_loss


if __name__ == "__main__":
    args = arg_parse()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    my_transforms = Compose([initialize_edge_weight])

    # Get X_datasets returns (datasets, names of datasets)
    test_datasets = get_chemical_datasets(my_transforms, -1, "test")
    test_datasets = [DataLoader(data, batch_size=64) for data in test_datasets[0]]

    val_datasets = get_chemical_datasets(my_transforms, -1, "val")
    val_datasets = [DataLoader(data, batch_size=64) for data in val_datasets[0]]

    # Setup wandb
    setup_wandb(vars(args), offline=False, name="noise-noise")


