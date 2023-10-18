import argparse
import logging
import random

import wandb
from datetime import datetime
from tqdm import tqdm
import os

import numpy as np
import torch
from ogb.graphproppred import Evaluator
from torch_geometric.transforms import Compose
from torch_scatter import scatter

from utils import setup_wandb
from datasets.loaders import get_train_loader, get_val_loaders, get_test_loaders


from unsupervised.embedding_evaluation import EmbeddingEvaluation, GeneralEmbeddingEvaluation, DummyEmbeddingEvaluation
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


def train_epoch(dataloader,
                model, model_optimizer,
                view_learner, view_optimizer,
                model_loss_all, view_loss_all, reg_all):
    """
    single train epoch for encoder and view learner

    Args:
        dataloader: dataloader (see get_train_loader)
        model: encoder
        model_optimizer: optimizer for model
        view_learner: view learner
        view_optimizer: optimizer for view learner
        model_loss_all: stores losses for each batch
        view_loss_all: stores view losses for each batch
        reg_all: stores regularization losses for each batch

    Returns:
        model_loss_all: loss for epoch
        view_loss_all: view loss for epoch
        reg_all: regularization loss for epoch
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for batch in tqdm(dataloader, leave = False):
        # set up
        batch = batch.to(device)

        # train view to maximize contrastive loss
        view_learner.train()
        view_learner.zero_grad()
        model.eval()

        x, _ = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, None)

        edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, batch.edge_attr)

        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + edge_logits) / temperature
        batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

        x_aug, _ = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, batch_aug_edge_weight)

        # regularization

        row, col = batch.edge_index
        edge_batch = batch.batch[row]
        edge_drop_out_prob = 1 - batch_aug_edge_weight

        uni, edge_batch_num = edge_batch.unique(return_counts=True)
        sum_pe = scatter(edge_drop_out_prob, edge_batch, reduce="sum")

        reg = []
        for b_id in range(args.batch_size):
            if b_id in uni:
                num_edges = edge_batch_num[uni.tolist().index(b_id)]
                reg.append(sum_pe[b_id] / num_edges)
            else:
                # means no edges in that graph. So don't include.
                pass
        num_graph_with_edges = len(reg)
        reg = torch.stack(reg)
        reg = reg.mean()

        view_loss = model.calc_loss(x, x_aug) - (args.reg_lambda * reg)
        view_loss_all += view_loss.item() * batch.num_graphs
        reg_all += reg.item()
        # gradient ascent formulation
        (-view_loss).backward()
        view_optimizer.step()

        # train (model) to minimize contrastive loss
        model.train()
        view_learner.eval()
        model.zero_grad()

        x, _ = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, None)
        edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, batch.edge_attr)

        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + edge_logits) / temperature
        batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze().detach()

        x_aug, _ = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, batch_aug_edge_weight)

        model_loss = model.calc_loss(x, x_aug)
        model_loss_all += model_loss.item() * batch.num_graphs

        # standard gradient descent formulation
        model_loss.backward()
        model_optimizer.step()

    return model_loss_all, view_loss_all, reg_all

def run(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info(args)
    # setup_seed(args.seed)

    if "original_datasets" not in os.listdir():
        os.mkdir("original_datasets")

    evaluation_node_features = args.node_features

    evaluator = Evaluator(name=args.dataset)
    my_transforms = Compose([initialize_edge_weight])


    dataloader = get_train_loader(args.batch_size, my_transforms)
    val_loaders, names = get_val_loaders(args.batch_size, my_transforms)
    test_loaders, names = get_test_loaders(args.batch_size, my_transforms)

    model = GInfoMinMax(MoleculeEncoder(emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
                    proj_hidden_dim=args.emb_dim).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)


    view_learner = ViewLearner(MoleculeEncoder(emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
                               mlp_edge_model_dim=args.mlp_edge_model_dim).to(device)
    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr)


    general_ee = GeneralEmbeddingEvaluation()

    model.eval()
    # general_ee.embedding_evaluation(model.encoder, val_loaders, names)
    # for ee in evaluators:
    # train_score, val_score, test_score = ee.embedding_evaluation(model.encoder, train_loader, valid_loader, test_loader)
    general_ee.embedding_evaluation(model.encoder, val_loaders, test_loaders, names, node_features = evaluation_node_features)
    # logging.info(
    #     "Before training Embedding Eval Scores: Train: {} Val: {} Test: {}".format(train_score, val_score,
    #                                                                                      test_score))

    model_losses = []
    view_losses = []
    view_regs = []
    valid_curve = []
    test_curve = []
    train_curve = []

    best_val = 1.0e4

    for epoch in tqdm(range(1, args.epochs + 1)):
        fin_model_loss = 0.
        fin_view_loss = 0.
        fin_reg = 0.

        model_loss_all = 0
        view_loss_all = 0
        reg_all = 0

        model_loss_all, view_loss_all, reg_all = train_epoch(dataloader,
                    model, model_optimizer,
                    view_learner, view_optimizer,
                    model_loss_all, view_loss_all, reg_all)

        fin_model_loss += model_loss_all / len(dataloader)
        fin_view_loss += view_loss_all / len(dataloader)
        fin_reg += reg_all / len(dataloader)

        model_losses.append(fin_model_loss)
        view_losses.append(fin_view_loss)
        view_regs.append(fin_reg)

        wandb.log({"Model Loss": fin_model_loss,
                   "View Loss": fin_view_loss,
                   "Reg Loss": fin_reg})

        if epoch % 2 == 0:
            model.eval()
            total_val = 0.
            total_train = 0.
            # general_ee.embedding_evaluation(model.encoder, val_loaders, names)
            # for ee in evaluators:
            # train_score, val_score, test_score = ee.embedding_evaluation(model.encoder, train_loader, valid_loader,
            #                                                              test_loader, vis=True)
            general_ee.embedding_evaluation(model.encoder, val_loaders, test_loaders, names, node_features = evaluation_node_features)
            # total_val += val_score
            # total_train += train_score


            # wandb.log({"Train Score": total_train,
            #            "Val Score": total_val})

            # train_curve.append(train_score)
            # valid_curve.append(val_score)
            # test_curve.append(test_score)

            # if total_val <= best_val:
            #     best_val = val_score
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': model.state_dict(),
                'encoder_optimizer_state_dict': model_optimizer.state_dict(),
                'view_state_dict': view_learner.state_dict(),
                'view_optimizer_state_dict': view_optimizer.state_dict()},
                f"{wandb.run.dir}/Checkpoint-{epoch}-{np.random.randint(0,10)}.pt")

    # train_score, val_score, test_score = ee.embedding_evaluation(model.encoder, train_loader, valid_loader,
    #                                                              test_loader, vis = True)

    general_ee.embedding_evaluation(model.encoder, val_loaders, test_loaders, names, node_features = evaluation_node_features)

    # if 'classification' in dataset.task_type:
    #     best_val_epoch = np.argmax(np.array(valid_curve))
    #     best_train = max(train_curve)
    # else:
    #     best_val_epoch = np.argmin(np.array(valid_curve))
    #     best_train = min(train_curve)
    #
    # logging.info('FinishedTraining!')
    # logging.info('BestEpoch: {}'.format(best_val_epoch))
    # logging.info('BestTrainScore: {}'.format(best_train))
    # logging.info('BestValidationScore: {}'.format(valid_curve[best_val_epoch]))
    # logging.info('FinalTestScore: {}'.format(test_curve[best_val_epoch]))

    # return valid_curve[best_val_epoch], test_curve[best_val_epoch]


def arg_parse():
    parser = argparse.ArgumentParser(description='AD-GCL ogbg-mol*')

    parser.add_argument('--dataset', type=str, default='ogbg-molesol',
                        help='Dataset')
    parser.add_argument('--model_lr', type=float, default=0.001,
                        help='Model Learning rate.')
    parser.add_argument('--view_lr', type=float, default=0.001,
                        help='View Learning rate.')
    parser.add_argument('--num_gc_layers', type=int, default=4,
                        help='Number of GNN layers before pooling')
    parser.add_argument('--pooling_type', type=str, default='standard',
                        help='GNN Pooling Type Standard/Layerwise')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimension')
    parser.add_argument('--mlp_edge_model_dim', type=int, default=128,
                        help='embedding dimension')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.,
                        help='Dropout Ratio / Probability')
    parser.add_argument('--epochs', type=int, default=128,
                        help='Train Epochs')
    parser.add_argument('--reg_lambda', type=float, default=5.0, help='View Learner Edge Perturb Regularization Strength')

    # parser.add_argument('--node-features', help="Whether to include node features in evaluation", action=argparse.BooleanOptionalAction)

    parser.add_argument(
        '-f',
        '--node-features',
        action='store_true',
        help='Whether to include node features (labels) in evaluation',
    )

    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()
    args = setup_wandb(args)
    run(args)

