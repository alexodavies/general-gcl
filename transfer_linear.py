import argparse
import concurrent.futures
import glob
import logging
import os
import random

import numpy as np
import torch
import yaml
import wandb

from torch_geometric.transforms import Compose

from utils import setup_wandb, wandb_cfg_to_actual_cfg
from datasets.loaders import get_val_loaders, get_test_loaders


from sklearn.metrics import roc_auc_score, mean_squared_error

from unsupervised.embedding_evaluation import GeneralEmbeddingEvaluation
from unsupervised.encoder import Encoder
from unsupervised.learning import GInfoMinMax
from unsupervised.utils import initialize_edge_weight
from unsupervised.view_learner import ViewLearner
from sklearn.linear_model import Ridge, LogisticRegression
from  sklearn.preprocessing import normalize

from torch.nn import MSELoss, BCELoss, Softmax, Sigmoid

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)



def tidy_labels(labels):
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

def get_targets(loader):
    # Iterate over items in loaders and convert to nx graphs, while removing selfloops
    data= []
    for batch in loader:
        data += batch.to_data_list()
    targets = []
    for idata in data:
        y = idata.y
        if len(y.shape) > 1:
            targets.append(y[0][0].item())
        else:
            targets.append(y.item())

    return tidy_labels(targets)

def get_datalist(loader):
    data= []
    for batch in loader:
        data += batch.to_data_list()
    return data

def python_index_to_latex(index: str):
    index = index.replace('e', r'\times 10^{')
    return index

def reindex(number1, number2):
    number_string1 = str('%.3g' % number1)
    number_string2 = str('%.3g' % number2)
    # number_string1 = python_index_to_latex('{:.2e}'.format(number_string1))
    try:
        num1_numerator, number_1_index = float(number_string1.split('e')[0]), int(number_string1.split('e')[1])
    except:
        num1_numerator = float(number_string1)
        number_1_index = 0

    try:
        num2_numerator, number_2_index = float(number_string2.split('e')[0]), int(number_string2.split('e')[1])
    except:
        num2_numerator = float(number_string2)
        number_2_index = 0
    # eg index = -3, number_2_index = -5, index_difference = -2
    index_difference = number_2_index - number_1_index
    if index_difference < -5:
        num2_numerator = 0
    else:
        num2_numerator = num2_numerator * (10**index_difference)
    # number_string2 = f'{numerator}e{number_1_index}'

    return num1_numerator, num2_numerator, number_1_index



def floats_to_pm(value, error):
    value, error, index = reindex(value, error)

    string_out = f"$({value} \\pm {error}) \\times 10" + r'{' + f"{index}" + r'}$'

    return string_out


def linear_validation(model, checkpoint_path, val_loader, test_loader, val_embeddings, test_embeddings, name = "blank"):
    # At the moment this is rigid to single-value predictions

    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = get_task_type(val_loader, name)

    model_name = checkpoint_path.split("/")[-1].split(".")[0]
    if model_name not in os.listdir("outputs"):
        os.mkdir(f"outputs/{model_name}")

    if task == "empty":
        print(f"Skipping {model_name}")
        return

    if task == "classification":
        loss_fn = BCELoss()
        score_fn = roc_auc_score
    else:
        loss_fn = MSELoss()
        score_fn = mean_squared_error


    if checkpoint_path != "untrained":
        model_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_dict['encoder_state_dict'], strict=False)
    model.to(device)

    val_targets = get_targets(val_loader)

    shuffle_inds = np.arange(len(val_targets))
    np.random.shuffle(shuffle_inds)
    val_targets = np.array(val_targets)[shuffle_inds]
    val_embeddings = val_embeddings[shuffle_inds, :]

    val_embeddings = normalize(val_embeddings)
    test_embeddings = normalize(test_embeddings)

    test_targets = get_targets(test_loader)

    num_val = val_targets.shape[0]
    num_test = len(test_targets)

    n_splits = 20
    n_per_split = int(num_val / n_splits)
    scores = []
    for n_split in range(n_splits):
        selection_indices = np.ones(num_val, dtype=bool)

        selection_indices[n_split*n_per_split:(n_split+1)*n_per_split] = 0
        split_targets = np.array(val_targets)[selection_indices]
        split_embeddings = val_embeddings[selection_indices]

        nan_indices = np.isnan(split_targets)
        split_embeddings = split_embeddings[~nan_indices]
        split_targets = split_targets[~nan_indices]

        if task == "classification":
            lin_model = LogisticRegression(dual=False, fit_intercept=True, max_iter = 5000)
        elif task == "regression":
            lin_model = Ridge(fit_intercept=True, copy_X=True, normalize = True) # LinearRegression()  # Ridge(fit_intercept=True, copy_X=True)

        lin_model.fit(split_embeddings, split_targets)
        pred = lin_model.predict(test_embeddings)
        scores += [score_fn(test_targets, pred)]

    print(f"{model_name} & {name} & {np.mean(scores)} & {np.std(scores)}")
    wandb.log({f"{name}/model-mean": np.mean(scores),
               f"{name}/model-dev": np.std(scores)})

def run(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info(args)
    wandb.log({"Transfer":True, "Linear_Transfer":True})
    # setup_seed(args.seed)

    num = args.num
    checkpoint = args.checkpoint
    evaluation_node_features = args.node_features

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
    wandb.log({"Model Name": model_name})

    # Get datasets
    my_transforms = Compose([initialize_edge_weight])

    test_loaders, names = get_test_loaders(args.batch_size, my_transforms, num=num)
    val_loaders, names = get_val_loaders(args.batch_size, my_transforms, num=2*num)
    model_name = checkpoint_path.split("/")[-1].split(".")[0]

    checkpoints = ["untrained", "social-100.pt", "chem-100.pt", "all-100.pt", "edge-views-all.pt"]
    # percentiles = [10., 10., 10., 0.5]
    for i_ax, checkpoint in enumerate(checkpoints):

        checkpoint = checkpoints[i_ax]

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
                    # print(wandb_cfg)
                except yaml.YAMLError as e:
                    # pass
                    print(e)

            args = wandb_cfg_to_actual_cfg(args, wandb_cfg)

        # Retrieved saved models and load weights

        model = GInfoMinMax(
            Encoder(emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio,
                    pooling_type=args.pooling_type),
            proj_hidden_dim=args.emb_dim).to(device)

        view_learner = ViewLearner(
            Encoder(emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio,
                    pooling_type=args.pooling_type),
            mlp_edge_model_dim=args.mlp_edge_model_dim).to(device)

        if checkpoint != "untrained":
            model_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(model_dict['encoder_state_dict'])
            view_learner.load_state_dict(model_dict['view_state_dict'])


        # Get embeddings
        general_ee = GeneralEmbeddingEvaluation()
        model.eval()
        val_all_embeddings, val_separate_embeddings = general_ee.get_embeddings(model.encoder, val_loaders, node_features=evaluation_node_features)
        test_all_embeddings, test_separate_embeddings = general_ee.get_embeddings(model.encoder, test_loaders, node_features=evaluation_node_features)

        for i_embedding, embedding in enumerate(val_separate_embeddings):
            val_loader = val_loaders[i_embedding]
            test_loader = test_loaders[i_embedding]
            test_embedding = test_separate_embeddings[i_embedding]
            name = names[i_embedding]
            linear_validation(model, checkpoint_path, val_loader, test_loader, embedding, test_embedding, name)

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
    parser.add_argument('--mlp_edge_model_dim', type=int, default=32,
                        help='embedding dimension')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.2,
                        help='Dropout Ratio / Probability')
    parser.add_argument('--epochs', type=int, default=5000,
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


    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    setup_wandb(args)
    run(args)

